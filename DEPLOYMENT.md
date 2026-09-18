# Public API and frontend access

The frontend calls the API over HTTPS. FastAPI connects to PostgreSQL inside
Docker. Browser code needs the API URL and each user's bearer token; it does not
need PostgreSQL credentials.

```text
Frontend browser → HTTPS :443 → Caddy → FastAPI :8000 → PostgreSQL :5432
```

This guide keeps the existing Compose deployment and ingested dataset. Run Compose
commands from the same directory and with the same project name used for ingestion,
so the existing database volume and file mount are reused. Update your existing
`.env`; do not replace it with `.env.example`.

## 1. Choose how the PC will be reached

For a direct connection, you need a domain such as `api.example.com` and a public
IP that routes to the server. Create a DNS A record for that public IPv4 address.
Only add an AAAA record if IPv6 also reaches the server. If the PC is behind a
router, forward TCP ports 80 and 443 to its LAN address and permit those ports
through the host/network firewall. Caddy must be able to bind those ports.

If the PC is behind carrier-grade NAT or a university network where inbound access
is unavailable, DNS alone will not expose it. Arrange an inbound route with the
network administrator, or use an outbound tunnel. Cloudflare Tunnel is one option;
when its agent runs on the host, route the public hostname to
`http://localhost:8000`. Review the provider's current request-size and timeout
limits against your NIfTI file sizes and synchronous ingestion time. Use a stable
hostname for collaboration rather than a temporary development tunnel.

### Outbound tunnel for a university/home network

Run `cloudflared` on the remote PC itself, alongside Docker. It connects outward;
your network must permit that connection (Cloudflare documents port 7844). Start
by confirming `curl http://localhost:8000/docs` works on that PC.

Install on Debian/Ubuntu using Cloudflare's repository:

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update
sudo apt-get install cloudflared
```

For an immediate development test without a domain or Cloudflare account:

```bash
cloudflared tunnel --url http://localhost:8000
```

The command prints a public HTTPS URL such as `https://random-name.trycloudflare.com`.
Keep the process running and open that URL with `/docs` appended from another
network. Share its `/api/v1` URL with the frontend developer. The hostname is
temporary and changes on restart. Quick Tunnels are for development, with no
uptime guarantee; they have a 200 concurrent request limit and no SSE support.
If you already have a `.cloudflared/config.yaml`, consult the Quick Tunnel docs
before using this command because an existing config can interfere.

For a stable public address:

1. Add a domain you control to Cloudflare, or coordinate with its administrator.
2. In the Cloudflare dashboard, open **Networking > Tunnels**, create a tunnel,
   and follow its Linux connector instructions on the remote PC. The installation
   command uses a tunnel token: `sudo cloudflared service install <TUNNEL_TOKEN>`.
3. Add a **Published application** route with hostname `api.your-domain.com` and
   Service URL `http://localhost:8000`.
4. Check `sudo systemctl status cloudflared --no-pager`, then test the hostname's
   `/docs` page from outside your network.

The token is a connector credential and is not shared with frontend developers.
With this route, Cloudflare provides the public HTTPS endpoint; skip the direct
Caddy setup below. The origin URL assumes `cloudflared` runs on the host; inside
a container, `localhost` refers to that container instead.

Cloudflare currently lists a **100 MB request upload limit on Free/Pro plans**.
The backend's 1 GiB upload setting cannot override a proxy's smaller limit, and
this backend does not implement chunked uploads. Large corrected masks may need
an alternative such as a university-operated public reverse proxy or private
VPN access. Keep large dataset ingestion on localhost/SSH when it exceeds proxy
request timeouts; the source dataset is already on the server.

## 2. Allow the frontend origins

Add the actual browser origins to the existing server `.env`:

```dotenv
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173","https://curator.example.com"]
```

Each origin includes scheme, hostname, and port where applicable, without a path
or trailing slash. `http://localhost:5173` means the frontend developer's own
browser origin. Add the frontend domain here, not merely the API domain. The
default is an empty list. Explicit bearer authentication is supported; cookie
authentication is not used. CORS controls browser access and does not grant user
permissions or replace JWT authentication.

Copy the updated application and Compose files to the server, preserving your
existing `volumes` configuration if you customized it. Then rebuild:

```bash
docker compose up -d --build backend
```

This does not require reingestion or a schema change. Never use `down --volumes`
on the existing deployment unless you intend to delete its database.

## 3. HTTPS for direct inbound access

Skip this section if you chose Cloudflare Tunnel above.

Install Caddy on the remote PC using its official Linux package instructions.
Set `/etc/caddy/Caddyfile` to the following, replacing the domain:

```caddyfile
api.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

This configuration assumes Caddy runs on the host, not in another container.
Use the published API port if you changed `API_PORT` from 8000. Keep the Compose
backend port bound to `127.0.0.1`; Caddy receives the internet traffic. PostgreSQL's
published port also remains bound to `127.0.0.1`.

Configure trusted forwarded headers so redirects preserve HTTPS. Set
`FORWARDED_ALLOW_IPS` to the address of the proxy as seen by Uvicorn; Docker port
forwarding may make this the Docker bridge gateway rather than `127.0.0.1`.
For a dedicated deployment where only the trusted proxy and trusted local
processes/containers can reach the backend, `FORWARDED_ALLOW_IPS=*` is an option.
Do not use that setting if untrusted clients can reach the backend directly.
Uvicorn reads this environment variable from Compose. Recreate the backend after
changing it:

```bash
docker compose up -d backend
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
sudo systemctl status caddy --no-pager
```

Caddy obtains and renews the HTTPS certificate when the domain and inbound routing
are configured correctly. Visit `https://api.example.com/docs` from outside the
server's network to verify public access. Dataset and file endpoints still require
authentication. An unauthenticated `/api/v1/auth/me` returning 401 is expected.

## 4. Give collaborators application accounts

Create one account per collaborator:

```bash
docker compose exec backend python -m app.cli developer1 \
  --full-name 'Frontend Developer' --role REVIEWER
```

Use ADMIN only for people who need to create and ingest datasets. REVIEWER users
can see all datasets/files and submit reviews/corrections; this MVP has no dataset
assignments or read-only application role. Public API reachability does not enable
anonymous access or self-registration. Docs and the OpenAPI schema are public;
case data remains behind login.

Share the following with frontend developers:

* Base URL: `https://api.example.com/api/v1`
* Documentation: `https://api.example.com/docs`
* OpenAPI schema: `https://api.example.com/openapi.json`
* Their individual application account.

Example browser JavaScript (use the person's entered credentials):

```javascript
const apiOrigin = "https://api.example.com";
const login = await fetch(`${apiOrigin}/api/v1/auth/login`, {
  method: "POST",
  body: new URLSearchParams({ username, password }),
});
if (!login.ok) throw new Error("Login failed");
const { access_token } = await login.json();

const response = await fetch(`${apiOrigin}/api/v1/datasets`, {
  headers: { Authorization: `Bearer ${access_token}` },
});
if (!response.ok) throw new Error(`API request failed: ${response.status}`);
const datasets = await response.json();
```

Use `POST /api/v1/auth/login` form fields and attach `Authorization: Bearer ...`
to subsequent API and file requests. Expired tokens require another login. File
URLs in API responses are relative to the API origin. For protected PNG/JPEG
display, fetch the file with the bearer header and create a blob URL; setting an
`<img src>` directly cannot attach this header. NIfTI/DICOM viewers also need to
send the token with their downloads. Range requests and relevant download
response headers are enabled in CORS. Do not embed shared credentials in a
frontend build.

## 5. Remote database access when SQL is needed

Frontend development normally uses the API. For SQL tools such as DBeaver or
pgAdmin, developers with server SSH access can use a tunnel from their own PC:

```bash
ssh -N -L 15432:127.0.0.1:5432 dprg-2@SERVER_SSH_ADDRESS
```

Use `localhost:15432` as the SQL client host/port and `curator` as the database.
The remote port must match `POSTGRES_PORT` if you changed its default. Give each
SQL user an appropriate database role; the application's `curator` role is a
superuser in this Compose setup and should not be shared as a frontend credential.

For example, create a read-only SQL account for curation records from the server:

```bash
docker compose exec postgres psql -U curator -d curator
```

```sql
CREATE ROLE curation_reader LOGIN;
\password curation_reader
GRANT CONNECT ON DATABASE curator TO curation_reader;
GRANT USAGE ON SCHEMA public TO curation_reader;
GRANT SELECT ON datasets, cases, annotation_versions, reviews TO curation_reader;
```

This deliberately omits the users table, which contains password hashes. It
does not create an application login. Direct public PostgreSQL is a separate
deployment choice requiring TLS, `pg_hba.conf` access rules, firewall rules, and
scoped database roles; it is unnecessary for browser API access.

## References

* [Caddy installation](https://caddyserver.com/docs/install)
* [Caddy automatic HTTPS](https://caddyserver.com/docs/automatic-https)
* [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/)
* [FastAPI behind a proxy](https://fastapi.tiangolo.com/advanced/behind-a-proxy/)
* [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/)
* [Cloudflare tunnel setup](https://developers.cloudflare.com/tunnel/get-started/)
* [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
* [Cloudflare request upload limits](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/4xx-client-error/error-413/)
* [PostgreSQL client access rules](https://www.postgresql.org/docs/17/auth-pg-hba-conf.html)
