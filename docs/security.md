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

## Trust decision — the nova-writer Kiro agent runs with full shell as `pollen`

The `nova-writer` Kiro agent (`config/kiro/nova-writer.json`, provisioned to
`~/.kiro/agents/nova-writer.json` by `scripts/install-device-units.sh`) is an
optional on-device writer engine: an agentic coding CLI (`kiro-cli`, ACP) that
authors `reachy_nova` skill code and behavior rules when invoked from the
forge/rules seams. Its tool config grants the **full** Kiro tool surface —
`read`, `write`, `shell`, `aws`, and the rest — with every one of those also
listed in `allowedTools`, and the standing ACP session it backs is started
with `--trust-all-tools`. In plain terms: **the agent has full shell access
as the `pollen` user on the CM4**, the same account the harness itself runs
as.

This is an explicit operator decision (Ori, 2026-08-19), not an oversight or
a default left un-tightened:

- **Architectural placement, not a tool allow-list.** The Kiro agent sits at
  what the build plan calls the "Nova 2 Lite boundary" — the same
  machine-enforced harness cognition tier that Nova 2 Lite's vision
  understanding already occupies (see `tests/test_harness_boundary.py`: no
  `reachy_mini` import, no `set_target`, AST-checked). The Kiro agent is a
  **cognition-level actor** newly added inside that boundary — it can reason
  about and author artifacts that flow back through sanctioned seams
  (`inject_text`, the forge pipeline, `rules_overlay`) — but it is never a
  second owner of the robot SDK. The full-shell grant is about what the
  writer process can do *on the device it already lives on*, not about
  granting it a second path onto the robot's motors/audio.
- **Blast radius is the `pollen` account on the device.** Full shell as
  `pollen` means the Kiro agent can do anything `pollen` can do on that CM4 —
  read/write any file `pollen` owns, run any command `pollen` can run. It
  does **not** grant root, does not grant a new network-reachable surface
  (see [Accepted residual exposure](#accepted-residual-exposure) above — this
  harness commits to opening no new listeners), and does not grant AWS
  credentials beyond whatever `pollen`'s environment already exposes to it.
  If the `pollen` account itself is compromised, this grant does not make
  that materially worse; if the Kiro agent itself misbehaves or is
  manipulated by a bad prompt, the practical ceiling on the damage is "what a
  bad `pollen` shell session could do," not "root on the robot" or "control
  of the physical robot outside sanctioned seams."
- **The code it *authors* is still sandboxed separately from the trust the
  agent itself runs with.** Full shell belongs to the Kiro **agent**, never
  to the `executor.py` artifacts it writes. Every forged skill the writer
  produces still has to pass `forge_validator`'s AST-only allow-list gate
  (import allow-list, forbidden names, no dunder access, call-target
  allow-list — see `docs/components/skill-forge.md`) before it is ever
  eligible to activate, and at runtime a validated skill only ever receives
  `ForgedSkillContext` — a seven-method surface (`ctx.gesture`,
  `ctx.vocalize`, `ctx.say`, `ctx.inject`, `ctx.state_get`,
  `ctx.state_update`, `ctx.emotion`) with no shell, no filesystem, no
  network. A trusted writer producing untrusted, sandboxed output is the
  same shape this harness already uses for the HTTP-backed forge path — Kiro
  changes who/where the authoring happens, not what the generated code is
  allowed to touch once it's live.

Practically: treat `--trust-all-tools` for `nova-writer` the same way you'd
treat giving a trusted human contributor an SSH login as `pollen` — it is a
real, non-trivial grant, scoped to one already-trusted account on one
already-owned device, made deliberately rather than by default.

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
