# Design browser: container, tunnel and runbook

A look-only browser for the three design roles. Headless Chromium runs in a rootless Docker container on the
**execution server**, which publishes its debugging port (CDP) on that server's 127.0.0.1:9222 only. One
restricted SSH tunnel, a systemd user unit on the **control server**, carries the control server's
127.0.0.1:9222 to it, and the design profiles there use `http://127.0.0.1:9222` as `design_browser.cdp_url`.

The repository installs none of this. The install below is the operator's work, and it waits for the release
of the design browser tool cards, which bring the design roles' `design_vision` tool.

| File | Runs on | What it is |
| --- | --- | --- |
| `browser.Dockerfile` | execution server | The image: the digest-pinned Playwright image of `scripts/journeys/images/browser.Dockerfile` plus socat. `watchdog.sh` is the main process; a `HEALTHCHECK` reads `/json/version`. |
| `watchdog.sh` | in the container | Starts Chromium and the port forward, checks both and exits when they fail. |
| `compose.yaml` | execution server | Builds and runs the image: `restart: unless-stopped`, publish on 127.0.0.1:9222 only, read-only root with tmpfs for the profile and `/tmp`, no volumes, `cap_drop: ALL`, `no-new-privileges`, 2 GB and 2 CPUs. |
| `design-browser-tunnel.service` | control server | The tunnel: `ssh -N -T -L 127.0.0.1:9222:127.0.0.1:9222` with its own restricted key. |

## How it recovers

Rootless Docker restarts a container only when it exits, so `watchdog.sh` exits instead of restarting Chromium
itself, and `restart: unless-stopped` starts the container again with a new Chromium and an empty profile.

- Every 15 s the watchdog checks that the Chromium and socat processes are alive; a dead one ends it with
  status 1.
- Every 60 s it reads `/json/version` with a 5 s timeout. While that fails it tries again every 15 s, and the
  third failure in a row ends it with status 1.
- After 6 hours without a client it ends with status 0, which recycles Chromium.

The image's `HEALTHCHECK` first runs 30 s after a start. A killed Chromium is therefore healthy again after
about 15 + 30 s. A frozen one is still running but stops answering, and is healthy again after at most about
105 + 30 s. Both are inside 3 minutes.

## Why the port is forwarded inside the container

Headless Chromium will not listen on a non-loopback address. This was tested outside Docker with Chrome for
Testing 149.0.7827.55 (Playwright's chromium-1228 build), started with the switches of `watchdog.sh` and
`--remote-debugging-address=0.0.0.0`, then with the container's own address. In both runs it printed
`DevTools listening on ws://127.0.0.1:9222/...`, its only listening socket was 127.0.0.1:9222, and
`/json/version` answered on 127.0.0.1 but not on the other address. Docker delivers a published port to the
container's own address, so `watchdog.sh` uses socat to forward that address's port 9222 to 127.0.0.1:9222.
socat cannot listen on 0.0.0.0:9222 instead, because Chromium already holds 127.0.0.1:9222
(`Address already in use`). From outside the container, the forward is reachable through the publish on the
execution server's 127.0.0.1:9222 and from the project's own Compose network, which has no other container.

Under the container rules Chromium needs two more things from `watchdog.sh`, and each was seen to fail without
them. The first is `--no-sandbox`: without capabilities, Chromium aborts with "No usable sandbox!". The second
is a writable `HOME`: with a read-only root, Chromium's crash handler aborts. `HOME` is `/tmp`, a tmpfs.

## Placeholders

| Placeholder | Meaning |
| --- | --- |
| `<execution-host>` | The execution server's host name, as the control server reaches it. |
| `<docker-user>` | The execution server account that runs rootless Docker and the container. |
| `<tunnel-user>` | The execution server account the tunnel logs in as. It holds only the tunnel key. |
| `<key-path>` | Absolute path of the tunnel's private key on the control server. |
| `<known-hosts-path>` | Absolute path of the tunnel's own known_hosts file on the control server. |
| `<design-profile-1>` … `<design-profile-3>` | The three design profiles. |
| `<board-slug>` | A kanban board on which the design roles may use the browser. |

The control server commands run as the account that runs hermes and the design profiles.

## Install on a fresh execution server

### 1. Rootless Docker with linger (execution server)

As an administrator, set up Docker's apt repository as described at
<https://docs.docker.com/engine/install/ubuntu/>. The commands below are for Ubuntu; on other distributions the
steps are the same with that distribution's packages.

```sh
# compose.yaml's CPU limit needs the cpu controller delegated to user services (cgroup v2).
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nDelegate=cpu cpuset io memory pids\n' | sudo tee /etc/systemd/system/user@.service.d/delegate.conf
sudo systemctl daemon-reload

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    docker-ce-rootless-extras uidmap dbus-user-session
# Only the rootless daemon may run: stop the system-wide one that the packages started.
sudo systemctl disable --now docker.service docker.socket
sudo rm -f /var/run/docker.sock

sudo useradd --create-home --shell /bin/bash <docker-user>
# Linger starts the account's user services, rootless Docker among them, at boot without a login.
sudo loginctl enable-linger <docker-user>
```

Then work as `<docker-user>`. `sudo` opens no login session, so first point the shell at the runtime directory
that linger keeps. Every later step as `<docker-user>` assumes these two exports.

```sh
sudo -iu <docker-user>
export XDG_RUNTIME_DIR=/run/user/$(id -u) DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
dockerd-rootless-setuptool.sh install
systemctl --user enable --now docker
docker info
```

`docker info` must list `rootless` under Security Options, show Cgroup Version 2 and print no `No cpu ...`
warning. On Ubuntu 24.04 and later the kernel restricts unprivileged user namespaces, and the setup tool may
stop for that reason. If it does, run the AppArmor commands it prints as an administrator, then run the tool
again.

### 2. Build and start the browser (execution server, as `<docker-user>`)

Copy this folder of the released repository to the execution server, for example to `~/design_browser`, and
keep `watchdog.sh` executable. Then run:

```sh
cd ~/design_browser
docker compose up -d --build
docker compose ps
```

The build pulls the pinned base image and Ubuntu's socat package. About 30 s after the start, `docker compose ps`
shows the service `Up ... (healthy)`.

### 3. The tunnel key

On the control server, create the key. It has no passphrase because the unit runs unattended. Its comment
marks the key line for the rollback. Use this key for nothing else.

```sh
ssh-keygen -t ed25519 -N '' -C design-browser-tunnel -f <key-path>
```

On the execution server, as an administrator, create the account the tunnel logs in as. Give it the key with
the restrictions below, putting the one line of `<key-path>.pub` in place of `<public key line>`:

```sh
sudo useradd --create-home --shell /usr/sbin/nologin <tunnel-user>
sudo install -d -m 700 -o <tunnel-user> -g <tunnel-user> /home/<tunnel-user>/.ssh
echo 'restrict,port-forwarding,permitopen="127.0.0.1:9222",command="/bin/false" <public key line>' \
    | sudo tee -a /home/<tunnel-user>/.ssh/authorized_keys
sudo chown <tunnel-user>: /home/<tunnel-user>/.ssh/authorized_keys
sudo chmod 600 /home/<tunnel-user>/.ssh/authorized_keys
```

Never put this key on `<docker-user>`. The key must allow port forwarding. On its own, that also allows `ssh -R`
listeners and forwards to Unix sockets, which `permitopen` does not limit. Step 4 refuses both, for
`<tunnel-user>` only. On `<docker-user>` the key would reach the rootless Docker socket, and with it every
container.

Back on the control server, write the execution server's host key into the tunnel's own known_hosts file:

```sh
ssh-keyscan -t ed25519 <execution-host> > <known-hosts-path>
ssh-keygen -lf <known-hosts-path>
```

Compare the printed fingerprint with the one the execution server shows for
`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`, read there through a channel you already trust. If they
differ, delete `<known-hosts-path>` and stop.

### 4. The tunnel account's sshd limits (execution server)

The key line limits where the key may connect, but not where it may listen or which Unix sockets it may reach.
On the execution server, as an administrator, refuse both for `<tunnel-user>` in an sshd drop-in, test the
configuration and read back what sshd will apply:

```sh
sudo tee /etc/ssh/sshd_config.d/design-browser-tunnel.conf <<'EOF'
Match User <tunnel-user>
    PermitListen none
    AllowStreamLocalForwarding no
EOF
sudo sshd -t
sudo sshd -T -C user=<tunnel-user>,host=localhost,addr=127.0.0.1 \
    | grep -E '^(permitlisten|allowstreamlocalforwarding) '
```

`PermitListen none` refuses every `ssh -R` listener, and `AllowStreamLocalForwarding no` refuses Unix-socket
forwarding in both directions. The key's one forward, to 127.0.0.1:9222, stays allowed. Ubuntu's `sshd_config`
includes this folder at its top and sshd keeps the first value it reads, so later settings do not override
these lines. The Match block ends at the end of the file.

`sshd -t` prints nothing. The read-back prints `permitlisten none` and `allowstreamlocalforwarding no`. That
read-back, not the file, shows that the limits apply. Only then reload sshd:

```sh
sudo systemctl reload ssh
```

If `sshd -t` prints an error or the read-back shows anything else, do not reload. Correct or delete the file and
run both again. Keep this administrator session open until check 3 passes: it is the way back in if sshd stops
accepting new connections.

### 5. The tunnel unit (control server)

From the repository checkout on the control server:

```sh
mkdir -p ~/.config/systemd/user
cp scripts/design_browser/design-browser-tunnel.service ~/.config/systemd/user/
# In the copy, replace <key-path>, <known-hosts-path>, <tunnel-user> and <execution-host>.
"${EDITOR:-vi}" ~/.config/systemd/user/design-browser-tunnel.service
systemctl --user daemon-reload
systemctl --user enable --now design-browser-tunnel.service
# Linger starts the unit at boot without a login.
sudo loginctl enable-linger "$USER"
systemctl --user status design-browser-tunnel.service
```

The status shows `active (running)`.

### 6. The design profiles (control server)

```sh
for profile in <design-profile-1> <design-profile-2> <design-profile-3>; do
  hermes -p "$profile" config set design_browser.cdp_url http://127.0.0.1:9222
  hermes -p "$profile" config set design_browser.boards '["<board-slug>"]'
done
```

List in `design_browser.boards` every board on which the design roles may use the browser, for example
`'["<board-slug-1>", "<board-slug-2>"]'`. This runbook does not set `design_browser.allowed_hosts`. While that
list is empty the design browser opens no page, and which hosts go there is the owner's decision.

## Checks

1. **9222 is bound only to 127.0.0.1.** On each server, `ss -Htln 'sport = :9222'` prints one line, with local
   address `127.0.0.1:9222`. On the execution server that is rootless Docker's port forwarder, on the control
   server the tunnel's ssh.
2. **`/json/version` answers through the tunnel.** On the control server,
   `curl -fsS http://127.0.0.1:9222/json/version` prints a JSON object with
   `"webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/..."`.
3. **The key opens no shell and no other port.** This check has three parts. Check 2 shows the other half: the
   same key's one forward, to 127.0.0.1:9222, still works.

   First, on the execution server, in the administrator session kept open since step 4:

   ```sh
   sudo -u <tunnel-user> python3 -c 'import socket; socket.socket(socket.AF_UNIX).connect("/run/dbus/system_bus_socket")'
   sock=$(mktemp -u /tmp/design-browser-check.XXXXXXXXXX)
   echo "sock=$sock"
   ```

   The first line prints nothing when `<tunnel-user>` can connect to the D-Bus system socket, which the fourth
   probe below tries. If it prints an error, check 3 fails. `mktemp -u` prints a name that no file has at that
   moment, and creates nothing. The last line prints `sock=` with that name.

   Then, on the control server, paste that `sock=` line into the shell and run:

   ```sh
   probe() {  # probe <exit status> <text> <ssh arguments ...>
     want=$1 text=$2
     shift 2
     out=$(timeout 15 ssh -i <key-path> -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
             -o UserKnownHostsFile=<known-hosts-path> -o ExitOnForwardFailure=yes -o LogLevel=VERBOSE "$@" \
             </dev/null 2>&1)
     code=$?
     if ! grep -q '^Authenticated to ' <<<"$out"; then reason="not logged in"
     elif ! grep -qF "$text" <<<"$out"; then reason="text not found"
     elif [ "$code" -ne "$want" ]; then reason="exit status not $want"
     else echo "pass: $*"; return
     fi
     printf 'FAIL: %s: %s, exit %s\n%s\n' "$*" "$reason" "$code" "$out"
   }
   probe 1 'This account is currently not available.' <tunnel-user>@<execution-host> id
   probe 255 'PTY allocation request failed' -tt <tunnel-user>@<execution-host>
   probe 255 'open failed: administratively prohibited' -W 127.0.0.1:22 <tunnel-user>@<execution-host>
   probe 255 'open failed: connect failed: open failed' -W /run/dbus/system_bus_socket <tunnel-user>@<execution-host>
   probe 255 'remote port forwarding failed for listen port 0' -N -R 127.0.0.1:0:127.0.0.1:9222 <tunnel-user>@<execution-host>
   probe 255 "remote port forwarding failed for listen path $sock" -N -R "$sock:127.0.0.1:9222" <tunnel-user>@<execution-host>
   ```

   Each probe runs ssh once and prints `pass` only when the key logged in, ssh printed the text and the exit
   status is the expected one. Otherwise it prints `FAIL` with the reason and the exit status, then all that ssh
   printed. At `LogLevel=VERBOSE` ssh prints `Authenticated to ...` once the key has logged in. An
   authentication, host key or network failure never prints it, though it also ends with exit 255. No probe
   reads the terminal, so `timeout` cannot leave the `-tt` probe stopped, and the block also works in a script.

   The texts are those of OpenSSH 9.6p1, the version of Ubuntu 24.04. On a server, `ssh -V` shows its version.

   - The first probe shows no `uid=` line. sshd runs the key's command through the account's login shell,
     `/usr/sbin/nologin`, which prints `This account is currently not available.` and exits 1. `/bin/false`
     would also exit 1.
   - The key's `restrict` refuses the second probe's terminal. ssh ends with exit 255 when a terminal that `-tt`
     forces is refused.
   - The key's `permitopen` refuses the third. A forward that sshd allowed but could not connect would say
     `connect failed` instead.
   - Step 4 refuses the fourth, and the key's `permitopen`, which holds no socket path, may refuse it as well.
     In this version sshd gives the same text for a socket it cannot reach. The first part shows that
     `<tunnel-user>` can reach this one.
   - Port 0 in the fifth asks for any free port, so a port in use cannot cause the refusal. Without step 4 sshd
     accepts it: ssh prints `Allocated port ... for remote forward to 127.0.0.1:9222`, keeps running, and
     `timeout` ends it with exit 124.
   - Without step 4 sshd accepts the sixth too. The fifth and the sixth are the probes that step 4 decides.

   Last, back on the execution server, in the same administrator session, where `sock` is still set:

   ```sh
   if sudo test -e "$sock"; then
     echo "FAIL: $sock exists"
     sudo rm -f "$sock"
   else
     echo "pass: nothing at $sock"
   fi
   ```

   A listener that sshd accepted leaves its socket file behind after ssh ends. A file that exists makes sshd
   refuse the listener whether or not step 4 applies. So the sixth probe's refusal counts only when its path was
   free before and is still free after. If anything exists there, check 3 fails, and `sudo rm -f` removes it. A
   retry starts again at the `mktemp` line, with a new path.

   Check 3 passes only when the D-Bus connect succeeds, all six probes print `pass`, and the path check prints
   `pass`.
4. **A killed and a frozen Chromium each come back healthy within 3 minutes.** On the execution server, as
   `<docker-user>`, in the compose folder:

   ```sh
   recover() {  # recover KILL | recover STOP
     id=$(docker compose ps -q design-browser)
     restarts=$(docker inspect -f '{{.RestartCount}}' "$id")
     browser=$(pgrep -P "$(docker inspect -f '{{.State.Pid}}' "$id")" -x chrome)
     kill -s "$1" "$browser"
     start=$SECONDS
     # Until Docker has started the container again, it still shows the old health: wait for "running" too.
     want="$((restarts + 1)) running healthy"
     until [ "$(docker inspect -f '{{.RestartCount}} {{.State.Status}} {{.State.Health.Status}}' "$id")" = "$want" ]; do
       if [ $((SECONDS - start)) -ge 180 ]; then echo "FAIL: not healthy within 3 minutes"; return 1; fi
       sleep 5
     done
     echo "healthy again after $((SECONDS - start)) s"
   }
   recover KILL
   recover STOP
   ```

   `KILL` kills Chromium's browser process and `STOP` freezes it. Each prints `healthy again after N s`, with
   N under 180: about 45 for KILL and up to about 140 for STOP. The watchdog writes its reason to
   `docker compose logs design-browser`.
5. **Both come back after a reboot.** Reboot the execution server. Once it is up, `docker compose ps` shows the
   service healthy again and check 2 passes again, because the tunnel retries every 5 s. Then reboot the
   control server. Check 2 passes again without anyone logging in.
6. **Each design role resolves design_vision and no general browser_* tool.** For each design profile, start
   `hermes -p <design-profile-N>` and enter `/tools`. The list has `design_vision` and no tool whose name
   starts with `browser_`.
7. **No other profile has design_browser keys.** On the control server,
   `grep -l '^design_browser:' ~/.hermes/config.yaml ~/.hermes/profiles/*/config.yaml` prints exactly the
   three design profiles' `config.yaml` files. If hermes uses another home, `hermes -p <profile> config path`
   shows a profile's file.
8. **No Chromium runs on the control server.** While a design role uses the browser,
   `pgrep -a -i -f 'chrom|headless_shell'` on the control server prints nothing.

## Rollback

1. Turn the browser off for the design roles (control server):

   ```sh
   for profile in <design-profile-1> <design-profile-2> <design-profile-3>; do
     hermes -p "$profile" config set design_browser.boards '[]'
   done
   ```

2. Remove the container (execution server, as `<docker-user>`, in the compose folder): `docker compose down`.
3. Remove the unit (control server):

   ```sh
   systemctl --user disable --now design-browser-tunnel.service
   rm ~/.config/systemd/user/design-browser-tunnel.service
   systemctl --user daemon-reload
   ```

4. Remove the key line (execution server):
   `sudo sed -i '/ design-browser-tunnel$/d' /home/<tunnel-user>/.ssh/authorized_keys`.
5. Remove the profile keys (control server). Check 7's `grep` then prints nothing.

   ```sh
   for profile in <design-profile-1> <design-profile-2> <design-profile-3>; do
     hermes -p "$profile" config unset design_browser.boards
     hermes -p "$profile" config unset design_browser.cdp_url
   done
   ```

The rollback leaves the following in place: rootless Docker and its linger, the built image, both accounts, the
sshd drop-in `/etc/ssh/sshd_config.d/design-browser-tunnel.conf`, the key files, the known_hosts file and the
copied folder. The drop-in only narrows what `<tunnel-user>` may do, and keeping it avoids another sshd reload.

## Known limits

- `docker compose stop` and `docker compose down` take 10 s. The watchdog does not handle SIGTERM, so Docker
  kills the container after its grace period.
- The browser container is given `shm_size` 1gb because the 64 MB Docker default crashes its graphics process on
  full-page screenshots. An older install picks up the change by running `docker compose up -d` from the design
  browser folder (`scripts/design_browser`).
