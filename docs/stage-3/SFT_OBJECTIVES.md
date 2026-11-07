# AYA Edge — Stage 3 SFT Objectives

## 1. Primary objective

Teach AYA Edge to follow instructions and produce useful, contextually appropriate responses.

## 2. General instruction following

AYA should learn to:

- Understand user requests.
- Follow explicit constraints.
- Answer direct questions.
- Produce requested formats.
- Avoid unnecessary responses.
- Ask for clarification when essential information is missing.
- Maintain consistency across related instructions.

## 3. Conversational behaviour

Training should include:

- Single-turn conversations.
- Multi-turn conversations.
- Follow-up questions.
- Context-dependent responses.
- Corrections.
- Requests for clarification.
- Summarisation.
- Natural conversational transitions.

## 4. Reasoning demonstrations

Introduce carefully selected demonstrations involving:

- Step-by-step problem solving.
- Logical relationships.
- Comparisons.
- Planning.
- Decision-making.
- Error identification.
- Explanations.

The objective is to improve task performance, not simply encourage longer answers.

Reasoning demonstrations must be checked for correctness.

## 5. Structured output generation

Train on structured formats including:

### JSON

Examples involving:

- Valid JSON syntax.
- Nested objects.
- Arrays.
- Required fields.
- Typed values.
- Schema compliance.

### XML

Examples involving:

- Valid XML structure.
- Correct nesting.
- Attributes.
- Required elements.
- Consistent formatting.

Evaluate both syntactic validity and semantic correctness.

## 6. Tool-use behaviour

Teach AYA to:

- Recognise when a tool is needed.
- Select an appropriate tool.
- Generate valid tool arguments.
- Respect tool schemas.
- Interpret tool results.
- Respond after tool execution.
- Handle tool failures.

Tool execution must be handled by a separate runtime.

## 7. Safety and uncertainty

Include examples demonstrating:

- Appropriate uncertainty.
- Avoidance of fabricated information.
- Handling of ambiguous requests.
- Safe refusal behaviour.
- Recognition of unsupported requests.
- Honest reporting of tool failures.

## 8. Success definition

Stage 3 succeeds when held-out evaluations demonstrate measurable improvement in instruction following, conversational quality, structured-output validity, and tool-call correctness.
