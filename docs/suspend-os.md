# Preventing OS Suspend for Reachy Mini

When the system suspends (sleep/hibernate), the `reachy-mini-daemon` stops running, cutting off communication with the robot. This guide explains how to disable and re-enable OS suspend.

## Disable Suspend (Keep Reachy Mini Running)

Mask all suspend/sleep targets so the system can never suspend:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

This blocks:
- Manual suspend (e.g. closing the lid)
- Idle-based suspend from GNOME power settings
- Any application attempting to trigger sleep/hibernate

## Enable Suspend (Restore Normal Behavior)

Unmask the targets to restore normal suspend behavior:

```bash
sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

## Verify Current Status

Check whether suspend targets are masked:

```bash
systemctl status sleep.target suspend.target hibernate.target hybrid-sleep.target | grep -E "Loaded|Active"
```

A masked target will show:
```
Loaded: masked (Reason: Unit sleep.target is masked.)
```

## USB Power (Bonus)

A udev rule is also in place at `/etc/udev/rules.d/90-reachy-mini-usb-power.rules` to ensure Reachy Mini USB devices are never autosuspended when reconnected:

- **Reachy Mini Audio** (`38fb:1001`)
- **Reachy Mini Serial/Control** (`1a86:55d3`)
- **Arducam 12MP Camera** (`0c45:636d`)
- **Terminus USB Hub** (`1a40:0101`)

To reload udev rules without rebooting:

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```
