# Call link and network setup

Call rooms accept an authenticated host and one invited guest. The host can share
the invitation URL shown in the call page; guests do not need a BridgeSign
account. Invitations expire after 24 hours by default and are signed with the
Flask `SECRET_KEY`. Local development creates an ephemeral key if `SECRET_KEY`
is unset; remote guest invitations are disabled with that temporary key. Set a
private, persistent `SECRET_KEY` before sharing public invitations or deploying.

For a public deployment, configure:

- `BRIDGESIGN_PUBLIC_URL`: the public HTTPS application origin, for example
  `https://sign.example.org`. The invitation link uses this origin when set.
- `BRIDGESIGN_CALL_INVITE_TTL_SECONDS`: optional invitation lifetime in seconds
  (default `86400`, minimum `60`).
- `BRIDGESIGN_TURN_ICE_SERVERS`: optional JSON array of TURN server settings,
  for networks where direct WebRTC connections cannot be established:

```json
[{
  "urls": "turns:relay.example.org:5349",
  "username": "short-lived-username",
  "credential": "short-lived-credential"
}]
```

TURN credentials are sent to call participants' browsers because WebRTC needs
them to use the relay. Use provider-issued, short-lived credentials; do not put
permanent relay secrets in this setting. Without TURN configuration, Call uses
public STUN servers only, which will not connect through every firewall or NAT.

The local development address `127.0.0.1` is only reachable from the same
computer. A copied localhost link cannot be used by someone on another device.
Remote camera and microphone access also requires HTTPS. The call page warns
when its generated link is local or non-HTTPS.
