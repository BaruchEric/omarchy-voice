#!/usr/bin/env bash
# Give omarchy-voice a pointer: install ydotool and let this user drive it.
#
# Hyprland can move the pointer but has no click dispatcher, so click_text and
# the wheel scroll go through ydotool, which writes to /dev/uinput. That node
# is root-only until udev says otherwise. Arch's package ships a user unit
# (ydotool.service) and a rule that opens the node to the `input` group; group
# membership only lands at the next login, so this also tags the node uaccess,
# which logind applies to the seated user right away.
#
#   sudo bash ~/.local/share/omarchy-voice/share/setup-click.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run me with sudo: sudo bash $0" >&2
  exit 1
fi
user=${SUDO_USER:-}
if [[ -z $user || $user == root ]]; then
  echo "run me through sudo from your own account, not as root directly" >&2
  exit 1
fi
uid=$(id -u "$user")

echo "== ydotool"
pacman -S --needed --noconfirm ydotool

echo "== /dev/uinput for $user"
usermod -aG input "$user"
# The package rule (80-uinput.rules) opens the node to the input group, but a
# desktop session that was logged in before that usermod does not carry the
# group, and neither does its systemd --user manager, which is what starts
# ydotoold. A uaccess tag makes logind put an ACL for the seated user on the
# node instead; it has to be set before 73-seat-late.rules applies the tags,
# hence the number.
rm -f /etc/udev/rules.d/80-uinput.rules
cat > /etc/udev/rules.d/71-uinput-uaccess.rules <<'RULE'
# omarchy-voice: ydotoold runs as the seated user, so the seated user gets uinput.
KERNEL=="uinput", TAG+="uaccess"
RULE
modprobe uinput 2>/dev/null || true
udevadm control --reload
# The node is a static one, so address it through sysfs; a bare name is
# refused with "Invalid argument" until the module has registered it.
udevadm trigger --action=add /sys/class/misc/uinput 2>/dev/null \
  || udevadm trigger --subsystem-match=misc --action=add
sleep 1
if getfacl -p /dev/uinput 2>/dev/null | grep -q "^user:$user:rw"; then
  echo "ok: logind granted $user /dev/uinput for this session"
else
  echo "warning: no ACL for $user on /dev/uinput; is a graphical session open on seat0?" >&2
fi

echo "== ydotoold as $user"
run_as_user() {
  runuser -u "$user" -- env XDG_RUNTIME_DIR="/run/user/$uid" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uid/bus" "$@"
}
run_as_user systemctl --user reset-failed ydotool.service 2>/dev/null || true
run_as_user systemctl --user enable --now ydotool.service
sleep 1
if [[ $(run_as_user systemctl --user is-active ydotool.service) == active ]] \
   && run_as_user test -S "/run/user/$uid/.ydotool_socket"; then
  echo "ok: ydotoold is up, socket at /run/user/$uid/.ydotool_socket"
else
  echo "warning: ydotoold did not start; see: journalctl --user -u ydotool.service" >&2
  echo "         logging out and back in picks up the input group as a fallback" >&2
  exit 1
fi
echo "done. omarchy-voice doctor lists ydotool under hands."
