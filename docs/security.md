# Security Hardening (P0)

This document covers the P0 security posture for the on-robot Reachy Nova
deployment (Reachy Mini Wireless, user `pollen`, harness checked out at
`~/git/reachy-nova`): scoping the IAM principal to exactly the Bedrock
models the harness invokes, locking down the `.env` file that carries AWS
credentials, keeping the nervous-system MQTT broker off the LAN, and
documenting the residual exposure we accept rather than pretend to fix.

Run `scripts/harden-robot.sh` to apply/verify the `.env` and mosquitto
checks below in one idempotent pass — see [Running the hardening
script](#running-the-hardening-script).

## IAM policy — scoped to exactly the invoked models

The harness invokes three Bedrock models (see `reachy_nova/config.py`):

| Service | Model ID | API used |
|---|---|---|
| Nova 2 Sonic (voice) | `amazon.nova-2-sonic-v1:0` | `bedrock:InvokeModelWithBidirectionalStream` |
| Nova 2 Lite (vision) | `us.amazon.nova-2-lite-v1:0` | `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream` |
| Nova multimodal embeddings (memory) | `amazon.nova-2-multimodal-embeddings-v1:0` | `bedrock:InvokeModel` |

The Lite model ID is `us.`-prefixed, which means it is a **cross-region
inference profile**, not a foundation model ARN directly. Bedrock routes
inference-profile invocations to one of the underlying regional foundation
models on your behalf, so the IAM policy must grant access to *both* the
inference-profile ARN the SDK calls and the underlying foundation-model ARNs
it fans out to (`us-east-1`, `us-east-2`, `us-west-2` for the `us.` profile
family) — granting only the profile ARN and denying the underlying models
produces an `AccessDeniedException` at invoke time even though the profile
ARN itself matches.

No other model, action, or resource is granted. Replace `<ACCOUNT_ID>` with
the robot's AWS account ID.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "NovaSonicBidirectionalStream",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModelWithBidirectionalStream",
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-sonic-v1:0"
    },
    {
      "Sid": "NovaLiteInferenceProfile",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:inference-profile/us.amazon.nova-2-lite-v1:0"
    },
    {
      "Sid": "NovaLiteUnderlyingFoundationModels",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-lite-v1:0",
        "arn:aws:bedrock:us-east-2::foundation-model/amazon.nova-2-lite-v1:0",
        "arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-2-lite-v1:0"
      ]
    },
    {
      "Sid": "NovaEmbeddings",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-multimodal-embeddings-v1:0"
    }
  ]
}
```

Attach this policy to a dedicated IAM user or role used only by the robot —
not a broad admin/PowerUser principal. Rotate the resulting access key the
same way any other credential is rotated; a scoped policy limits blast
radius but does not replace rotation.

Honesty condition: `aws bedrock invoke-model-with-bidirectional-stream`,
`invoke-model`, and `invoke-model-with-response-stream` against these three
model IDs succeed under the attached policy; any other model ID or Bedrock
action is denied.

## `.env` permissions

The `.env` at `~/git/reachy-nova/.env` carries `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` in plaintext. A probe of the robot found it shipped
`-rw-rw-r--` — world-readable, so any local user or process on the robot
could read live AWS credentials.

Fix and verify:

```bash
chmod 600 ~/git/reachy-nova/.env
ls -l ~/git/reachy-nova/.env
# expect: -rw------- 1 pollen pollen ... .env
```

`scripts/harden-robot.sh` performs and re-verifies this chmod on every run
and exits non-zero if the mode is anything other than `-rw-------` (600)
afterward.

## mosquitto — localhost-only

`config/mosquitto/mosquitto.conf` sets `allow_anonymous true` — there is no
authentication layer on the broker, so the only thing standing between the
LAN and the nervous-system event bus is the bind address. The broker must
listen on `127.0.0.1` only, never on `0.0.0.0` or a LAN interface.

Verify with:

```bash
ss -tlnp | grep ':1883'
# expect a line whose local address is 127.0.0.1:1883 (or [::1]:1883),
# never 0.0.0.0:1883 or *:1883
```

`scripts/harden-robot.sh` runs this same check: mosquitto bound to anything
other than loopback on `:1883` is a hard failure naming the finding;
mosquitto not running at all is a warning, not an error (it means nothing is
exposed, not that hardening failed).

## Accepted residual exposure

The ReachyMiniOS daemon and Zenoh are upstream realities this harness does
not own and does not attempt to close in this pass:

- **ReachyMiniOS daemon, `:8000`** — LAN-open, unauthenticated. Ships this
  way from Pollen Robotics upstream; closing it is out of scope for the
  Reachy Nova harness.
- **Zenoh, `:7447`** — LAN-open, unauthenticated. Same upstream reality —
  Zenoh is the SDK's own transport, not something this harness starts or
  configures.

What this harness commits to instead: it **opens no new network listeners**
beyond the mosquitto broker above, which is verified loopback-only. A LAN
port scan of the robot (e.g. `nmap -p- <robot-ip>`) run before and after
deploying this harness must find the same listeners plus nothing new — any
newly-appearing LAN-reachable port introduced by this harness is a
regression, not an accepted residual.

## Running the hardening script

```bash
scripts/harden-robot.sh                    # defaults to ~/git/reachy-nova/.env
scripts/harden-robot.sh /path/to/other.env # explicit .env path
```

Idempotent and safe to re-run on every deploy or cron tick; requires no
`sudo`. It chmods and verifies the `.env`, verifies the mosquitto bind
address, and prints the residual-exposure summary above. A non-zero exit
means one of the two checks found a real finding — see the `FAIL:` line for
which one.
