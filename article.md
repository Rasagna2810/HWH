# How Hindsight Changed the Next Recommendation After a Failed Integration

The first recommendation an agent makes is not the interesting part. The interesting part is what happens after that recommendation fails, and whether the next run knows enough to avoid making the same suggestion again.

I built AegisAI around that problem. It analyzes customer accounts, gathers internal and external context, proposes a next action, and records what the customer-success team actually did with that recommendation. The part I ended up caring about most was the boundary between storing history and using history: MongoDB remains the operational source of truth, while Hindsight gives the planner a memory of decisions and outcomes that can influence what it recommends next.

## The system I built

The application has a React dashboard in front of a FastAPI workflow. The backend coordinates several specialized pieces rather than asking one model to do everything.

At a high level, the request path looks like this:

```text
Customer Success User
        |
   React Dashboard
        |
     FastAPI
        |
     Planner
      /   \
  Hindsight  Gemini
        |
   analysis strategy
        |
  Knowledge Agent ---> Qdrant
        |
 External Research
        |
 Recommendation / Report
```

The planner is the entry point for the decision process. It looks at the customer account and the available interaction history, then determines what kind of analysis is required. Internal knowledge is retrieved through Qdrant, external research can be added when needed, and the final workflow produces an actionable recommendation.

MongoDB has a different job. It stores customer records, interactions, recommendation records, approvals, feedback, and operational state. I did not want an agent-memory system to become a replacement for the application's transactional database.

That distinction matters.

If I need to answer "what is Meridian Healthcare's current contract status?", I want a database query. If I need the planner to answer "what happened the last time we took an action like this, and what was the outcome?", that is where persistent agent memory becomes useful.

That is the role I gave [Hindsight GitHub](https://github.com/vectorize-io/hindsight).

## The problem with remembering decisions

A recommendation without an outcome is only half a record.

Suppose a customer has an Epic EHR integration problem affecting 12 departments. The planner might reasonably recommend validating the integration across all 12 departments.

A human CSM accepts the recommendation and the engineering team works on it.

Then reality happens.

Ten departments pass validation. Two do not because of firewall configuration.

If the system only remembers the original recommendation, the next analysis still sees something like:

```text
Recommendation:
Validate the Epic integration across all 12 departments.
```

That is not enough information to improve the next decision.

The useful memory is closer to:

```text
Action:
Validate Epic integration across all 12 departments.

Outcome:
10 departments passed.
2 departments failed because of firewall issues.

Implication:
The remaining work is firewall remediation, not another full validation cycle.
```

That distinction shaped how I integrated Hindsight. I don't retain only the recommendation text. The memory lifecycle includes decisions, approvals or rejections, feedback, and completed outcomes.

The repository explicitly models this split: recommendation decisions and completed outcomes are retained under stable document IDs, with customer and industry tags. The planner recalls those decision memories before choosing its analysis strategy.

## Why I kept MongoDB in the picture

It was tempting to treat Hindsight as the database for everything. I deliberately didn't.

The application already has operational entities: customers, contacts, interactions, recommendations, tasks, and outcomes. Those records have application-level relationships and lifecycle rules. MongoDB is a much better fit for that state.

Hindsight answers a different question.

The [Hindsight documentation](https://hindsight.vectorize.io/) describes the basic model as retaining information, recalling relevant memories, and using those memories during agent reasoning. That maps neatly onto the lifecycle I needed:

```text
Operational event
      |
      v
MongoDB
      |
      +----> Hindsight retain
                 |
                 v
          Long-term memory
                 |
                 v
          Planner recall
                 |
                 v
          Next decision
```

This also gives me a failure boundary.

The application can continue operating from MongoDB history when Hindsight is unavailable. Hindsight is enabled through configuration rather than being a hard dependency of every request.

The configuration is intentionally small:

```env
HINDSIGHT_BASE_URL=http://localhost:8888
HINDSIGHT_API_KEY=
HINDSIGHT_BANK_ID=aegisai-customer-success
HINDSIGHT_TIMEOUT_SECONDS=30
```

The important detail here is not the number of environment variables. It is that memory infrastructure has an explicit boundary. If the memory service disappears, I don't want customer records or recommendation state to disappear with it.

## The integration is small, but the lifecycle matters

The backend uses the official `hindsight-client` Python SDK. Because the FastAPI workflow is asynchronous, the integration uses the async APIs: `aretain()` for writing memory and `arecall()` for retrieving it.

Conceptually, the write path looks like this:

```python
await hindsight.aretain(
    bank_id=HINDSIGHT_BANK_ID,
    content=outcome_content,
    document_id=document_id,
    tags=[customer_name, industry],
)
```

And the planner's read path is the corresponding recall:

```python
memories = await hindsight.arecall(
    bank_id=HINDSIGHT_BANK_ID,
    query=customer_query,
)
```

The important part is not the SDK call itself. It is what goes into `content`, how it is identified, and when it is written.

A memory saying "recommendation approved" is weak.

A memory saying "recommendation X was approved, completed, and resulted in Y" is much more useful when the planner later evaluates a similar situation.

That is why I treat completion and feedback as first-class memory events rather than merely logging them.

## Stable document IDs solved a less obvious problem

Long-term memory becomes difficult when the same logical event can be written repeatedly.

Retries happen. A user clicks twice. A request times out after the remote service accepted it. A workflow is resumed. If every retry creates an unrelated memory, the agent can end up seeing several copies of the same decision.

I therefore use stable document IDs for recommendation decisions and completed outcomes.

The design is roughly:

```text
Recommendation
    |
    +--> stable document ID
    |
    +--> customer tag
    |
    +--> industry tag
    |
    +--> decision / outcome content
```

The stable identifier gives the memory layer a durable reference to the application event instead of treating every network attempt as a new piece of history.

The tags provide another useful boundary. A planner working on Meridian should retrieve memories relevant to Meridian rather than mixing them with an unrelated logistics customer that happened to receive a similar recommendation.

This is one of the areas where I found the distinction between ordinary retrieval and agent memory useful. [Vectorize's explanation of agent memory](https://vectorize.io/what-is-agent-memory) describes memory as persistent context that can influence later agent behavior. In this system, that means preserving the relationship between a decision and what happened afterward, not just indexing a pile of customer text.

## The failed integration is where the design becomes useful

Consider the Meridian account again.

The current account state says the Epic integration is a serious problem. Recent customer interactions say the customer needs a recovery plan and production validation.

The planner proposes:

```text
Validate the Epic integration across all 12 departments.
```

The CSM approves it.

Later, the outcome is recorded:

```text
10 departments passed validation.
2 departments failed because of firewall configuration.
```

That outcome is retained.

On the next analysis, the planner has two categories of information available.

The first is current operational state from MongoDB:

```text
Customer: Meridian Healthcare
Issue: Epic integration
Status: Renewal risk
```

The second is accumulated decision context from Hindsight:

```text
Previous action: validate all departments
Result: 10 passed, 2 failed
Known blocker: firewall configuration
```

Now the next recommendation has a much more useful starting point.

Instead of treating "Epic integration problem" as a new problem, the planner can reason from the recorded outcome and focus on the unresolved firewall issue.

That is the behavior I wanted from memory.

I wasn't trying to make the model "remember the conversation." I wanted it to remember **what we tried and what happened**.

## Rejection is also useful memory

I initially thought successful actions would be the most valuable memories.

They aren't always.

A rejected recommendation contains information about what the CSM did not consider appropriate. For example:

```text
Recommendation:
Schedule a broad customer training session.

Decision:
Rejected.

Feedback:
Customer is blocked by the integration issue; training is premature.

Outcome:
Focus on integration recovery first.
```

If the planner later encounters the same account state, that rejection is useful context.

The repository therefore treats approval, rejection, feedback, and completion as memory events rather than only retaining successful outcomes.

That makes the feedback loop look like this:

```text
Recommendation
      |
      +---- Approved ----> Action ----> Outcome
      |
      +---- Rejected ----> Feedback
                              |
                              v
                         Hindsight
                              |
                              v
                       Future planning
```

The point is not that every future decision must follow the previous decision. Quite the opposite: the memory gives the planner evidence about what happened before, while the current account state still determines what needs attention now.

## Hindsight is not my RAG layer

There is another architectural distinction I found important.

AegisAI already uses Qdrant for internal knowledge retrieval. That is useful for questions such as:

```text
What does our integration runbook say about Epic deployments?
```

Hindsight is useful for a different class of question:

```text
What happened the last time we took an action on this customer's integration problem?
```

Those questions may use similar words, but they represent different data.

Qdrant is primarily retrieving relevant knowledge from a knowledge base.

Hindsight is preserving experience from the agent's own interactions and decisions.

That difference is reflected directly in the architecture:

```text
Internal knowledge
      |
    Qdrant
      |
"What does the documentation say?"

Customer decisions and outcomes
      |
   Hindsight
      |
"What happened when we tried this?"
```

The [Hindsight project](https://github.com/vectorize-io/hindsight) provides the memory primitives for that second path rather than forcing me to build an application-specific history retrieval system from scratch.

## What I learned building the feedback loop

### 1. Store outcomes, not just decisions

A recommendation is an intention. An outcome is evidence.

If memory only contains what the agent suggested, future runs know what the agent believed but not whether that belief survived contact with reality.

That makes outcome capture more important than simply increasing the amount of historical text.

### 2. Memory needs application-level identity

A memory system does not automatically understand my application's notion of identity.

I had to make customer and industry context explicit and keep stable document IDs for logical recommendation events.

Otherwise, "similar" memories can become misleading memories.

### 3. Keep operational state separate from agent memory

MongoDB remains the system of record for application state.

Hindsight adds a memory layer around decisions and experience.

That separation makes the architecture easier to reason about and gives me a fallback when memory infrastructure is unavailable.

### 4. Rejections are data

I don't want the agent to remember only things that worked.

A rejected recommendation tells me something about the account, the timing, or the CSM's judgment. That information can be just as useful as a successful action.

### 5. The hard part is the feedback boundary

Calling `retain()` and `recall()` is straightforward.

The harder engineering question is deciding **when an event has become meaningful enough to remember** and **what exactly should be remembered**.

If I retain every intermediate model response, memory becomes noisy. If I retain only final recommendations, I lose the outcome that explains whether the recommendation was useful.

The useful boundary in this system is the decision lifecycle: recommendation, human decision, action, feedback, and outcome.

## Where this leaves the architecture

The resulting system is deliberately less magical than the phrase "AI that learns" suggests.

The planner does not rewrite its own weights. There is no hidden training loop after every customer interaction.

Instead, I built a persistent feedback path:

```text
Current customer state
        |
        v
     Planner
        |
        v
 Recommendation
        |
        v
 Human decision
        |
        v
 Action + outcome
        |
        v
    Hindsight
        |
        v
 Future planner recall
        |
        v
 Better-informed next decision
```

That is a much more concrete definition of learning for an agent system.

The model can still make a bad recommendation. Hindsight does not guarantee correctness. What changes is that the next decision no longer has to start with a blank memory of what happened before.

For customer-success workflows, that distinction matters. The valuable context is often not another document about the customer. It is the history of **what we tried, what the customer accepted, what failed, and what happened afterward**.

That is the part I wanted the system to remember.
