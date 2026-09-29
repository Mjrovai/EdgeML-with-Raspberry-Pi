#!/bin/bash
# Writes a NetworkManager Wi-Fi connection onto the Orange Pi SD card (ext4) from macOS.
# Usage: ./setup-wifi.sh <partition>   e.g. ./setup-wifi.sh /dev/disk4s1  (find it with: diskutil list)
# Requires Homebrew e2fsprogs (brew install e2fsprogs). Tested with Orangepizero3w_1.0.2_debian_trixie.
set -euo pipefail

DEV="${1:-}"
if [ -z "$DEV" ]; then
  echo "Usage: $(basename "$0") <partition>   e.g. $(basename "$0") /dev/disk4s1  (find it with: diskutil list)"; exit 1
fi
SBIN="$(brew --prefix e2fsprogs 2>/dev/null)/sbin"
[ -x "$SBIN/debugfs" ] || { echo "debugfs not found: run 'brew install e2fsprogs' first."; exit 1; }
NAME=wifi-ssh.nmconnection
DIR=/etc/NetworkManager/system-connections

# Use sudo only when the target is not writable (a raw device needs root; a test image does not)
SUDO=""; [ -w "$DEV" ] || SUDO=sudo

# Safety check: only write to the Orange Pi root partition (label opi_root)
LABEL=$($SUDO "$SBIN/e2label" "$DEV" 2>/dev/null || true)
if [ "$LABEL" != "opi_root" ]; then
  echo "$DEV does not look like the Orange Pi root partition (label: '${LABEL:-none}', expected 'opi_root'). Aborting."
  exit 1
fi

read -r -p "Wi-Fi network name (SSID): " SSID
read -r -s -p "Wi-Fi password: " PSK; echo
[ ${#PSK} -ge 8 ] || { echo "WPA password must be at least 8 characters."; exit 1; }

diskutil unmountDisk "${DEV%s[0-9]*}" >/dev/null 2>&1 || true

echo "Checking the filesystem..."
$SUDO "$SBIN/e2fsck" -p "$DEV" || { echo "e2fsck reported problems; aborting."; exit 1; }

TMP=$(mktemp "${TMPDIR:-/tmp}/nm.XXXXXX"); chmod 600 "$TMP"
trap 'rm -P "$TMP" 2>/dev/null || rm -f "$TMP"' EXIT

esc() { printf '%s' "$1" | sed 's/\\/\\\\/g'; }
cat > "$TMP" <<EOF
[connection]
id=$(esc "$SSID")
uuid=$(uuidgen | tr 'A-Z' 'a-z')
type=wifi
autoconnect=true

[wifi]
mode=infrastructure
ssid=$(esc "$SSID")

[wifi-security]
key-mgmt=wpa-psk
psk=$(esc "$PSK")

[ipv4]
method=auto

[ipv6]
method=auto
EOF

echo "Writing $DIR/$NAME ..."
$SUDO "$SBIN/debugfs" -w "$DEV" -f - >/dev/null 2>&1 <<EOF
cd $DIR
rm $NAME
write $TMP $NAME
sif $NAME mode 0100600
sif $NAME uid 0
sif $NAME gid 0
EOF

echo "Result:"
$SUDO "$SBIN/debugfs" -R "ls -l $DIR" "$DEV" 2>/dev/null | grep "$NAME" \
  || { echo "File not found on the card - something went wrong."; exit 1; }
$SUDO "$SBIN/debugfs" -R "cat $DIR/$NAME" "$DEV" 2>/dev/null | sed 's/^psk=.*/psk=********/'

if diskutil eject "${DEV%s[0-9]*}" >/dev/null 2>&1; then
  echo "Card ejected. Put it in the Orange Pi and power it on."
fi
