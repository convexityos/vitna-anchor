import test from "node:test";
import assert from "node:assert/strict";

import {
  compileJsonSchemaToPda,
  createGrammarParser,
  formatGrammarSummary,
  PDA_STATES,
} from "../runtime/grammar.mjs";

test("compileJsonSchemaToPda extracts required properties and rules", () => {
  const schema = {
    type: "object",
    properties: {
      action: { type: "string" },
      iterations: { type: "number" },
      active: { type: "boolean" },
    },
    required: ["action", "active"],
  };

  const pda = compileJsonSchemaToPda(schema);
  assert.equal(pda.schemaType, "object");
  assert.equal(pda.propertyCount, 3);
  assert.equal(pda.requiredCount, 2);
  assert.equal(pda.initialState, PDA_STATES.EXPECT_OBJ_OPEN);

  const summary = formatGrammarSummary(pda);
  assert.ok(summary.includes("PUSHDOWN GRAMMAR PDA COMPILER"));
  assert.ok(summary.includes("action"));
  assert.ok(summary.includes("[REQUIRED]"));
  assert.ok(summary.includes("[OPTIONAL]"));
});

test("createGrammarParser validates syntactic tokens and structural correctness", () => {
  const schema = {
    type: "object",
    properties: {
      name: { type: "string" },
    },
    required: ["name"],
  };

  const parser = createGrammarParser(schema);

  // Valid token stream
  const r1 = parser.feedToken('{"name"');
  assert.equal(r1.accepted, true);

  const r2 = parser.feedToken(':"anchor"}');
  assert.equal(r2.accepted, true);
  assert.equal(parser.isAccepted(), true);
  assert.deepEqual(parser.populatedKeys, ["name"]);

  // Invalid token stream test
  const invalidParser = createGrammarParser(schema);
  const badRes = invalidParser.feedToken("InvalidNonJson");
  assert.equal(badRes.accepted, false);
});
