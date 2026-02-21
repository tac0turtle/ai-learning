# 12 Factor Agents

Source: [HumanLayer / Dex](https://github.com/humanlayer/12-factor-agents)

Core thesis: **Production agents are mostly deterministic software with LLM steps sprinkled in at just the right points.** The naive "here's your prompt, here's a bag of tools, loop until done" pattern doesn't work for production customers. Frameworks get you to 70-80% quality, then you hit a wall and start reverse-engineering their internals. These principles let you build modular, production-grade LLM software without that trap.

---

## The Meta-Insight

Most "AI agents" in production are not that agentic. They're carefully designed software with strategic LLM decision points. The fastest path to shipping quality AI products is not adopting a framework wholesale -- it's taking small, modular agent concepts and incorporating them into existing products incrementally.

The journey most founders go through: pick framework -> get to 80% -> realize 80% isn't enough -> reverse-engineer the framework -> start over. These factors let you skip that cycle.

---

## Factor 1: Natural Language to Tool Calls

The foundational pattern. Convert human intent into structured function calls that deterministic code can execute.

```
"create a payment link for $750 to Terri" -> { function: "create_payment_link", parameters: { amount: 750, customer: "cust_..." } }
```

**The small thing:** The LLM doesn't need to make the API call. It just decides *what* to call. Your code picks up the structured output and does the actual execution. This separation is key -- you control validation, auth, retries, error handling. The LLM is just translating intent.

**The small thing:** A real agent doing this wouldn't have the customer ID in the prompt. It would need to list customers, list products, list prices -- multiple tool calls to build the payload. Or you pre-fetch those IDs into the context window (see Factor 13).

---

## Factor 2: Own Your Prompts

Don't outsource prompt engineering to a framework's black box.

Framework style (bad for production):
```python
agent = Agent(role="...", goal="...", personality="...", tools=[...])
```

This hides the actual tokens hitting the model. You can't tune what you can't see.

Instead, treat prompts as first-class code. Write the actual system/user messages yourself. Template them with your data.

**The small thing:** "Role hacking" -- you can abuse user/assistant role boundaries in ways framework abstractions won't let you. Some of the best performance tricks involve non-standard role usage.

**The small thing:** You want the flexibility to try *everything*. Different prompt structures, different role assignments, different instruction formats. A framework locks you into its abstraction.

**The small thing:** Build tests and evals for your prompts just like you would for regular code. Prompts are code now.

---

## Factor 3: Own Your Context Window

**This is the most important factor.** Everything is context engineering. LLMs are stateless functions that turn inputs into outputs. Better inputs = better outputs.

Context includes: prompts, instructions, RAG documents, history, tool calls, memory, retrieved data.

At any given point, your input to the LLM is: **"here's what's happened so far, what's the next step?"**

**The small thing:** You don't have to use the standard `[{role: "system", content: "..."}, {role: "user", ...}]` message format. You can pack everything into a single user message with XML-tagged sections:

```xml
<slack_message>
    From: @alex
    Channel: #deployments
    Text: Can you deploy the backend?
</slack_message>

<list_git_tags_result>
    tags:
      - name: "v1.2.3"
        commit: "abc123"
</list_git_tags_result>
```

This can be more token-efficient and attention-efficient than the standard format.

**The small thing:** Build your own context serialization. A `Thread` with `Event[]` where each event has a type and data. Custom `event_to_prompt()` and `thread_to_prompt()` functions give you total control over how history is represented.

**The small thing:** You can hide resolved errors from the context window. If a tool failed then succeeded on retry, strip the failed attempt -- it's noise now.

**The small thing:** Filter sensitive data *before* it enters context. Session IDs, passwords, API keys -- minimize what the LLM sees.

**The small thing:** Same information, fewer tokens. Compact representations (YAML vs verbose JSON, stripped keys vs full objects) directly affect quality within the attention window.

---

## Factor 4: Tools Are Just Structured Outputs

Reframe: "tool calling" is just the LLM outputting JSON that your code interprets. The LLM decides *what*, your code decides *how*.

```python
class CreateIssue:
    intent: "create_issue"
    issue: Issue

class SearchIssues:
    intent: "search_issues"
    query: str
    what_youre_looking_for: str
```

**The small thing:** Just because an LLM "called a tool" doesn't mean you have to execute a corresponding function in the same way every time. You can intercept, modify, gate, or redirect based on business logic.

**The small thing:** The "next step" might not be atomic. A tool call output might trigger a multi-step deterministic workflow, not just one function.

**The small thing:** There's a high-stakes first-token choice happening: is the model returning plaintext content, or structured JSON? Having the model *always* output JSON (including for "done" or "need more info" intents) removes this ambiguity and can improve reliability.

---

## Factor 5: Unify Execution State and Business State

Don't maintain separate tracking for "where am I in the workflow" (execution state) vs "what has happened" (business state). They diverge and create bugs.

Instead: infer execution state from the event history. If the last event is a `deploy_backend_result` with status "success", you know the deploy happened. No need for a separate `current_step = "post_deploy"` variable.

**The small thing:** The thread/event list IS the state. It's trivially serializable, debuggable, forkable, and resumable. You can copy a thread into a new context to "fork" an agent's state.

**The small thing:** Minimize out-of-band state. Session IDs, auth tokens -- these might need to live outside the thread, but your goal is to keep them minimal. Everything else should be derivable from the event history.

**The small thing:** This makes debugging radically simpler. The entire history is in one place, one data structure, one source of truth.

---

## Factor 6: Launch/Pause/Resume with Simple APIs

Agents are programs. You need to start, stop, query, and resume them through simple interfaces.

**The small thing:** Most frameworks let you pause/resume, but NOT between tool selection and tool execution. This is the critical gap. If the LLM says "deploy to production" and you can't pause before actually deploying, you're stuck with: (a) keeping the process alive in memory while waiting for approval, (b) restricting to low-stakes tools only, or (c) YOLO.

**The small thing:** External triggers (webhooks, crons, other agents) should be able to resume an agent without deep integration with the orchestrator. Save state -> break loop -> webhook comes in -> load state -> continue.

**The small thing:** When combined with unified state (Factor 5), pause/resume becomes trivial: serialize the thread, store it, load it later, keep going.

---

## Factor 7: Contact Humans with Tool Calls

Make human interaction a first-class tool, not a special case.

```python
class RequestHumanInput:
    intent: "request_human_input"
    question: str
    context: str
    options: Options  # urgency, format (free_text, yes_no, multiple_choice), choices
```

When the agent needs human input, it outputs this structured JSON like any other tool call. Your code saves state, notifies the human, breaks the loop. When the human responds (via webhook), load state and continue.

**The small thing:** This enables "outer loop" agents -- agents that are triggered by crons/events/other agents, not by a human chatting. The agent works for 5, 20, 90 minutes, and when it hits a decision point, it reaches out to a human. The flow is Agent->Human, not the typical Human->Agent.

**The small thing:** The same abstraction extends to Agent->Agent communication trivially.

**The small thing:** Urgency/format metadata on the human request lets the notification system make smart decisions about how to interrupt the human (Slack DM vs email vs SMS).

---

## Factor 8: Own Your Control Flow

Don't delegate control flow to the framework. Build your own while loop with explicit handling for each intent type.

```python
while True:
    next_step = await determine_next_step(thread)

    if next_step.intent == 'request_clarification':
        # async: break loop, wait for webhook
        await send_message_to_human(next_step)
        break
    elif next_step.intent == 'fetch_open_issues':
        # sync: execute immediately, continue loop
        issues = await linear_client.issues()
        thread.events.append(result)
        continue
    elif next_step.intent == 'create_issue':
        # high-stakes: break loop, wait for approval
        await request_human_approval(next_step)
        break
```

**The small thing:** Different tool calls warrant different control flow. Some are fire-and-continue (fetching data). Some are break-and-wait (human approval). Some are break-and-sleep (long-running pipelines). You need to decide this per-tool, not hand it to a generic loop.

**The small thing:** This is where you inject: summarization of tool results, LLM-as-judge on outputs, context compaction, logging/tracing/metrics, client-side rate limiting, durable sleep.

**The small thing:** The switch/match statement IS your agent. It's just code. You can unit test it, debug it, add logging, handle edge cases -- all with normal software engineering practices.

---

## Factor 9: Compact Errors into Context Window

When a tool call fails, feed the error back into context so the LLM can self-heal.

**The small thing:** Use a consecutive error counter. Cap retries at ~3 per tool. Without this, the agent will spin out repeating the same failing call.

```python
consecutive_errors = 0
# ...
except Exception as e:
    consecutive_errors += 1
    if consecutive_errors < 3:
        thread.events.append({"type": "error", "data": format_error(e)})
    else:
        break  # escalate to human, reset context, or give up
```

**The small thing:** `format_error(e)` -- don't dump raw stack traces. Distill the error into actionable context. The LLM doesn't need 50 lines of traceback; it needs "Failed to connect to deployment service: connection refused on port 8080."

**The small thing:** Hitting the error threshold is a great place to escalate to a human (Factor 7), either by model decision or deterministic takeover.

**The small thing:** You can restructure how errors are represented, remove previous failed events from context, or rewrite the error narrative entirely. Own your context (Factor 3) applies to errors too.

---

## Factor 10: Small, Focused Agents

**As context grows, LLMs lose focus.** Keep agents to 3-10 steps, maybe 20 max. Each agent has a narrow, well-defined scope.

**The small thing:** This is true even if LLMs get dramatically smarter. Larger context windows becoming reliable just means each small agent can handle a slightly larger chunk of the DAG. You expand scope incrementally as capability grows, not by building monoliths.

**The small thing:** From the NotebookLM team: "the most magical moments come when I'm really close to the edge of the model capability." Finding that boundary and operating precisely at it is a moat. Small focused agents let you find and ride that edge.

**The small thing:** Small agents are independently testable. You can eval one agent without needing the entire pipeline to work. This is essential for iteration speed.

**The small thing:** This is also the #1 defense against error spin-outs (Factor 9). A focused agent with a small context window is far less likely to get confused and loop on the same error.

---

## Factor 11: Trigger from Anywhere, Meet Users Where They Are

Agents should be invocable from Slack, email, SMS, webhooks, crons, other agents, scheduled jobs -- whatever.

**The small thing:** This enables "outer loop" agents -- agents kicked off by events, not humans. They work autonomously and reach out to humans only at critical decision points.

**The small thing:** If you can quickly loop in a variety of humans through their preferred channel, you can give agents access to higher-stakes operations. The approval mechanism (Factor 7) becomes the safety net that unlocks more powerful agent actions.

---

## Factor 12: Make Your Agent a Stateless Reducer

`(state, event) -> new_state`

Your agent is a pure function. Given the current thread state and a new event, it produces updated state. No hidden side effects, no implicit state.

**The small thing:** This is the same mental model as Redux, event sourcing, or a left fold. If you've built systems with those patterns, agent architecture will feel natural.

**The small thing:** Stateless reducers are trivially testable: give it state X and event Y, assert it produces state Z. No mocking, no setup, no teardown.

**The small thing:** This is what ties all the other factors together. Unified state (5) + owned control flow (8) + stateless computation = an agent you can reason about like any other piece of software.

---

## Factor 13 (Appendix): Pre-fetch Context You'll Need

If there's a high probability the model will call tool X, don't waste a round trip. Call it deterministically and put the result in context before the LLM even runs.

```python
# Instead of: let the LLM ask for git tags, then fetch, then continue
# Do: fetch git tags up front, include in context
git_tags = await fetch_git_tags()
thread.events.append({"type": "list_git_tags_result", "data": git_tags})
next_step = await determine_next_step(thread)
```

**The small thing:** This eliminates an entire LLM round trip. If you know the agent will need customer data, deployment status, or git tags -- just fetch them. Let the LLM do the *hard* part (reasoning about the data), not the trivial part (deciding to fetch it).

**The small thing:** You can remove the tool from the available tools entirely, simplifying the LLM's decision space.

---

## Recurring Themes

1. **You want flexibility to try EVERYTHING.** The best approach for your use case will be discovered through experimentation, not prescribed by a framework.

2. **Agents are just software.** Normal engineering practices (testing, debugging, logging, explicit control flow) apply directly.

3. **Context engineering is the game.** Not prompt engineering, not fine-tuning, not model selection. What goes into the context window and how it's formatted determines output quality.

4. **Deterministic where possible, LLM where necessary.** Don't use the LLM to make decisions that your code already knows the answer to. Use it for the genuinely ambiguous, language-understanding, reasoning-required parts.

5. **Incremental adoption over framework adoption.** Pick the factors that solve your immediate problems. You don't need all 12 on day one.
