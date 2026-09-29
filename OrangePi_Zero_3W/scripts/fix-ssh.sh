#!/bin/bash
# Diagnoses and fixes a missing SSH server on the Orange Pi SD card (ext4) from macOS.
# Adds a systemd drop-in that runs "ssh-keygen -A" (creates missing host keys) before sshd starts.
# Usage: ./fix-ssh.sh <partition>   e.g. ./fix-ssh.sh /dev/disk4s1  (find it with: diskutil list)
# Requires Homebrew e2fsprogs (brew install e2fsprogs). Tested with Orangepizero3w_1.0.2_debian_trixie.
set -euo pipefail

DEV="${1:-}"
if [ -z "$DEV" ]; then
  echo "Usage: $(basename "$0") <partition>   e.g. $(basename "$0") /dev/disk4s1  (find it with: diskutil list)"; exit 1
fi
SBIN="$(brew --prefix e2fsprogs 2>/dev/null)/sbin"
[ -x "$SBIN/debugfs" ] || { echo "debugfs not found: run 'brew install e2fsprogs' first."; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
REPORT="$HERE/opi-diagnostics.txt"
DROPDIR=/etc/systemd/system/ssh.service.d
DROPIN=regen-host-keys.conf

SUDO=""; [ -w "$DEV" ] || SUDO=sudo
dbg() { $SUDO "$SBIN/debugfs" -R "$1" "$DEV" 2>/dev/null; }

# Safety check: only write to the Orange Pi root partition (label opi_root)
LABEL=$($SUDO "$SBIN/e2label" "$DEV" 2>/dev/null || true)
if [ "$LABEL" != "opi_root" ]; then
  echo "$DEV does not look like the Orange Pi root partition (label: '${LABEL:-none}', expected 'opi_root'). Aborting."
  exit 1
fi

diskutil unmountDisk "${DEV%s[0-9]*}" >/dev/null 2>&1 || true

echo "Checking the filesystem..."
$SUDO "$SBIN/e2fsck" -p "$DEV" || { echo "e2fsck reported problems; aborting."; exit 1; }

echo "Collecting diagnostics into $REPORT ..."
{
  echo "=== /etc/ssh"; dbg "ls -l /etc/ssh"
  echo "=== sshd_not_to_be_run?"; dbg "stat /etc/ssh/sshd_not_to_be_run" | head -1 || true
  echo "=== sshd / firstrun lines (syslog, auth.log)"
  for f in /var/log/syslog /var/log/auth.log; do
    dbg "cat $f" | grep -aiE "sshd|ssh\.service|firstrun|host ?key" | tail -40 || true
  done
  echo "=== HDMI / display lines (kern.log)"
  dbg "cat /var/log/kern.log" | grep -aiE "hdmi|drm|disp|edid" | tail -40 || true
  echo "=== zero-length files under /etc"
  RD=$(mktemp -d "${TMPDIR:-/tmp}/etc.XXXXXX")
  $SUDO "$SBIN/debugfs" -R "rdump /etc $RD" "$DEV" >/dev/null 2>&1 || true
  (cd "$RD" && $SUDO find etc -type f -size 0 | sort)
  $SUDO rm -rf "$RD"
} > "$REPORT" 2>&1

# Empty host keys are not "missing" for ssh-keygen -A, so delete them now
EMPTY_KEYS=$(dbg "ls -l /etc/ssh" | awk '$NF ~ /^ssh_host_/ && $6 == 0 {print $NF}')

TMP=$(mktemp "${TMPDIR:-/tmp}/dropin.XXXXXX")
trap 'rm -f "$TMP"' EXIT
cat > "$TMP" <<'EOF'
[Service]
ExecStartPre=
ExecStartPre=/bin/sh -c 'find /etc/ssh -maxdepth 1 -name "ssh_host_*" -size 0 -delete; /usr/bin/ssh-keygen -A'
ExecStartPre=/usr/sbin/sshd -t
EOF

echo "Writing $DROPDIR/$DROPIN ..."
$SUDO "$SBIN/debugfs" -w "$DEV" -f - >/dev/null 2>&1 <<EOF
mkdir $DROPDIR
sif $DROPDIR mode 040755
sif $DROPDIR uid 0
sif $DROPDIR gid 0
cd $DROPDIR
rm $DROPIN
write $TMP $DROPIN
sif $DROPIN mode 0100644
sif $DROPIN uid 0
sif $DROPIN gid 0
rm /etc/ssh/sshd_not_to_be_run
$(for k in $EMPTY_KEYS; do echo "rm /etc/ssh/$k"; done)
EOF

echo "Result:"
dbg "ls -l $DROPDIR" | grep "$DROPIN" || { echo "Drop-in not found on the card - something went wrong."; exit 1; }
dbg "cat $DROPDIR/$DROPIN"
echo; echo "Empty SSH host keys removed: ${EMPTY_KEYS:-none}" | tr '\n' ' '; echo
echo "SSH host keys left on the card:"; dbg "ls -l /etc/ssh" | awk '$NF ~ /^ssh_host_/ {print "  " $NF, $6 " bytes"}'

if diskutil eject "${DEV%s[0-9]*}" >/dev/null 2>&1; then
  echo "Card ejected. Put it in the Orange Pi and power it on."
fi
