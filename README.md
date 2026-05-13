# claude-code-langfuse-hook

See what your team is doing in Claude Code.

This is a small tool that hooks into Claude Code and sends every turn
to [Langfuse](https://langfuse.com/) — so you can see who used Claude,
on which project, how many tokens, and how much it cost.

You get:

- One trace per turn, grouped by session.
- Tagged with the project name and model.
- Attributed to the engineer's git email.
- Token counts (including prompt-cache hits) so Langfuse can show real cost.

---

## Before you start a couple of disclaimers

**This works with Claude Code CLI only.** It does *not* work with
Claude on the web ([claude.ai](https://claude.ai/)) or the Claude
desktop app. The desktop app runs in its own sandbox and doesn't
expose hooks, so there's no way for this package to plug in.

**The costs are estimates.** Numbers are computed from token counts
using Anthropic's published API base rates. They won't match your
official Anthropic invoice down to the cent. Treat them as a way to
get a feel for usage by team, project, or person — not as a billing
source of truth.

---

## Step 1 — Get a Langfuse instance

You need a Langfuse somewhere to send the data to. Three options:

- **Try it on your laptop** — run Langfuse locally with Docker Compose:
[langfuse.com/self-hosting/local](https://langfuse.com/self-hosting/local).
- **Self-host on a VM** — deploy on any VM or Kubernetes cluster:
[langfuse.com/self-hosting](https://langfuse.com/self-hosting).
- **Use the cloud** — sign up at [cloud.langfuse.com](https://cloud.langfuse.com/).

Once it's running, create a project inside Langfuse and copy the
**public key**, **secret key**, and **base URL**. You'll paste them in
below.

---

## Step 2 — Install the hook

```bash
pipx install claude-code-langfuse-hook   # recommended
# or
pip install --user claude-code-langfuse-hook
```

Then plug it into Claude Code:

```bash
claude-langfuse install
```

This adds one line to `~/.claude/settings.json` so every Claude Code
session triggers the hook. You only do this once per machine.

---

## Step 3 — Turn it on for a project

Add five lines to your project's `.env` file:

```env
CC_TRACE_TO_LANGFUSE=true
CC_PROJECT_NAME=my-project
CC_LANGFUSE_BASE_URL=https://langfuse.example.com
CC_LANGFUSE_PUBLIC_KEY=pk-lf-...
CC_LANGFUSE_SECRET_KEY=sk-lf-...
```

Don't want to type them? Run `claude-langfuse init` and copy-paste.

That's all. The next time you use Claude Code in this project, traces
will show up in Langfuse.

> **Tip:** real OS environment variables beat the `.env` file. So you
> can commit the non-secret lines to `.env.example` and let direnv,
> doppler, 1Password, or Vault inject the keys.

> **Note on the `CC_` prefix:** every variable starts with `CC_` so
> this hook doesn't accidentally pick up keys meant for some other
> Langfuse-using app running in the same shell. Plain
> `LANGFUSE_PUBLIC_KEY` is ignored on purpose.

---

## Commands you can run


| Command                     | What it does                                  |
| --------------------------- | --------------------------------------------- |
| `claude-langfuse install`   | Hooks into Claude Code (one time per machine) |
| `claude-langfuse uninstall` | Unhooks it                                    |
| `claude-langfuse init`      | Prints the `.env` block to copy-paste         |
| `claude-langfuse status`    | Tells you if everything is wired up           |
| `claude-langfuse test`      | Sends a fake trace to check the connection    |
| `claude-langfuse hook`      | Internal — Claude Code runs this for you      |


---

## What shows up in Langfuse

For every Claude Code turn, you'll see one trace shaped like this:

```
trace          "Claude Code - Turn N"     session, user, project, model
└─ span        "Claude Code - Turn N"     the user's prompt
   ├─ generation  "Claude Response"       Claude's reply + token usage
   └─ tool        "Tool: <name>"          one per tool Claude called
```

A few things to know:

- **Who did it.** `user_id` is taken from `git config user.email`. If
that's missing, the hook falls back to your OS username and logs a
warning.
- **Grouping.** All turns from the same Claude Code session share a
`session_id`, so they group nicely in Langfuse's Sessions view.
- **Big payloads get trimmed.** Anything over `CC_LANGFUSE_MAX_CHARS`
(default 20,000) is truncated. The trim records the original length
and a sha256 hash so you can prove the original existed.

### Token usage and cost

Claude charges four different rates depending on what kind of input
tokens you used — especially when prompt caching is on. The hook
sends all four to Langfuse:


| Token kind                    | Roughly costs    |
| ----------------------------- | ---------------- |
| `input_tokens`                | base input       |
| `output_tokens`               | base output      |
| `cache_creation_input_tokens` | 1.25× base input |
| `cache_read_input_tokens`     | 0.1× base input  |


Langfuse's built-in price table already knows these names, so you'll
see accurate dollar amounts without any extra setup.

### Extended thinking

When Claude is in "extended thinking" mode, the reasoning isn't shown
in the main input/output (which would be noisy). Instead it's tucked
into the generation's **Metadata → thinking** tab in Langfuse so you
can read it if you want to.

---

## All the variables


| Var                      | Required? | What it does                                           |
| ------------------------ | --------- | ------------------------------------------------------ |
| `CC_TRACE_TO_LANGFUSE`   | yes       | Set to `true` to turn tracing on for this project      |
| `CC_PROJECT_NAME`        | yes       | The project name tagged on every trace                 |
| `CC_LANGFUSE_BASE_URL`   | yes       | Your Langfuse URL                                      |
| `CC_LANGFUSE_PUBLIC_KEY` | yes       | Langfuse public key                                    |
| `CC_LANGFUSE_SECRET_KEY` | yes       | Langfuse secret key                                    |
| `CC_LANGFUSE_DEBUG`      | no        | Set to `true` for chatty logs                          |
| `CC_LANGFUSE_MAX_CHARS`  | no        | Trim long content above this size (default 20000)      |
| `CC_TRACE_SUBAGENTS`     | no        | Set to `true` to also trace subagents (off by default) |


---

## Things that might go wrong


| What you see                                  | Why                                         | What to do                                                                                                                   |
| --------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| No traces in Langfuse                         | The hook isn't installed                    | Run `claude-langfuse install`, then `claude-langfuse status` to confirm                                                      |
| `status` says config is incomplete            | A required variable is missing              | Run `claude-langfuse init` and fill in the five `CC_`* vars                                                                  |
| `user_id` is your OS username, not your email | Git doesn't know your email                 | `git config user.email you@example.com`                                                                                      |
| Tool calls show up as plain spans             | Your Langfuse is older than v3.10           | Upgrade Langfuse                                                                                                             |
| Hook fails with `socksio` import error        | You use a SOCKS proxy                       | `pipx inject claude-code-langfuse-hook httpx[socks]`                                                                         |
| Cost is $0                                    | Langfuse doesn't have a price for the model | In Langfuse: Settings → Models → add the model with prices for `input`, `output`, `cache_creation_input`, `cache_read_input` |
| Hook times out (over 15s)                     | Your transcript is huge                     | Raise the `timeout` in `~/.claude/settings.json`, or lower `CC_LANGFUSE_MAX_CHARS`                                           |


---

## Working on this package

```bash
git clone <repo>
cd pypi-package
pip install -e . pytest
pytest
```

---

## License

MIT.