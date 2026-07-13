---
name: vastai-ssh-runpod-image-fix
description: "Vast.ai + runpod/* image SSH 'Permission denied (publickey)' fix — /root is group-writable so sshd StrictModes refuses the key; fix via --onstart chmod go-w /root"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---

**Symptom:** Create a Vast.ai instance with `--image runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2404 --ssh --direct`, key shows as attached in vast UI ("Instance SSH Keys"), `ssh-keygen -y -f ~/.ssh/id_ed25519` matches the registered pubkey, yet BOTH direct and proxy SSH give `Permission denied (publickey)`.

**Root cause (from `vastai logs <id>`):**
```
Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys
Failed publickey for root ... ED25519 SHA256:<my key>   <- key IS matched
```
The key is present and matched, but sshd `StrictModes` rejects it because the runpod image ships **`/root` group/other-writable**. sshd checks the WHOLE path (`/root`, `/root/.ssh`, the file) — a writable `/root` fails StrictModes even when the file is 600.

**Why earlier attempts failed:**
- `vastai attach ssh` / vast account keys → populate the PROXY, not this image's container sshd.
- `vastai execute <id> '<cmd>'` → returns `400 Invalid command given` on this image (unusable).
- An onstart that only did `chmod 700 /root/.ssh; chmod 600 authorized_keys` → still failed because `/root` itself was writable.

**THE FIX — pass an `--onstart` script that fixes the whole path** (runs alongside the image's sshd, does NOT replace the entrypoint so sshd stays up):
```bash
mkdir -p /root/.ssh
echo 'ssh-ed25519 AAAA... bharat-vast.ai' >> /root/.ssh/authorized_keys
chown -R root:root /root/.ssh
chmod go-w /root            # <- THE missing piece
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
echo "StrictModes no" >> /etc/ssh/sshd_config   # belt-and-suspenders
(service ssh reload || kill -HUP "$(pgrep -of sshd)") 2>/dev/null || true
```
Working file: `scripts/onstart_addkey.sh`. Create with `--onstart scripts/onstart_addkey.sh`.

**Connect:** use the DIRECT route with the explicit key + `-o IdentitiesOnly=yes`:
`ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p <HostPort-of-22/tcp> root@<public_ipaddr>`
Get IP/port from `vastai show instance <id> --raw` → `public_ipaddr` + `ports['22/tcp'][0]['HostPort']`.

**Also (separate gotcha):** do NOT pack ssh flags into a shell var like `KEY="-i ~/.ssh/id_ed25519 -o ..."` then `ssh $KEY` — word-splitting mangles it ("Identity file ... not accessible"). Use explicit flags inline each call.

**Cost of NOT knowing this:** ~3 pod create/destroy cycles, ~$2-3 + 30 min wasted (2026-06-06). See [[vastai_h200_workflow]] (the vast DEFAULT image injects keys fine via proxy; only runpod/* images need this).

---
**2026-06-07 — Vast.ai `stop` does NOT preserve an instance reliably.** Stopped B200
39752999 overnight; next day `vastai start` → `404 Instance not found` and `vastai show
instances` was empty — the host RECLAIMED the stopped instance (unlike RunPod where stop
preserves). Lesson: on Vast.ai, a stopped instance can be deleted by the host at any time.
Copy ALL needed artifacts (adapters, data, logs) to local BEFORE stopping; assume a stopped
pod may be gone. Don't rely on "stop to resume tomorrow" for Vast.ai.
