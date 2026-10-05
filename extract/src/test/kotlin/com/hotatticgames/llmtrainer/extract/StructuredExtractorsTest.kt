package com.hotatticgames.llmtrainer.extract

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class StructuredExtractorsTest {
    private fun run(name: String, s: String) = Extractors.extract(name, null, s.toByteArray(Charsets.UTF_8))

    @Test fun csvQuotingEmbeddedNewlinesAndOffsets() {
        val csv = "Part,Spec,Note\r\n\"Caliper, front\",25 Nm,\"say \"\"hi\"\"\nsecond line\"\r\n\r\nPad,10 Nm,\r\n"
        val e = run("t.csv", csv)
        assertEquals(2, e.blocks.size)
        assertEquals("Part: Caliper, front; Spec: 25 Nm; Note: say \"hi\"\nsecond line", e.blocks[0].text)
        assertEquals("Part: Pad; Spec: 10 Nm", e.blocks[1].text)
        assertEquals("record", e.blocks[0].kind)
        assertEquals("\"Caliper, front\",25 Nm,\"say \"\"hi\"\"\nsecond line\"", e.docText.substring(e.blocks[0].charStart, e.blocks[0].charEnd))
        assertEquals("Pad,10 Nm,", e.docText.substring(e.blocks[1].charStart, e.blocks[1].charEnd))
        assertTrue(e.issues.none { it.severity == "error" })
    }

    @Test fun tsvAndRaggedRows() {
        val e = run("t.tsv", "a\tb\n1\t2\n3\n4\t5\t6\n")
        assertEquals(3, e.blocks.size)
        assertEquals("a: 4; b: 5; col3: 6", e.blocks[2].text)
        assertEquals("ragged_rows", e.issues.single().code)
    }

    @Test fun csvHeaderOnlyIsEmptyContent() {
        assertEquals("empty_content", run("t.csv", "a,b,c\n").issues.single().code)
    }

    @Test fun csvUnterminatedQuoteWarns() {
        val e = run("t.csv", "a,b\n1,\"oops\n2,3\n")
        assertTrue(e.issues.any { it.code == "corrupt_structured" && it.severity == "warning" })
        assertEquals(1, e.blocks.size)
    }

    @Test fun jsonArrayRecordsWithExactSpansAndOrder() {
        val j = "[ {\"zeta\": \"last?\", \"alpha\": 5, \"nested\": {\"k\": [1, \"x\"]}},\n  {\"alpha\": \"second\", \"e\": \"\"} ]"
        val e = run("d.json", j)
        assertEquals(2, e.blocks.size)
        assertEquals("zeta: last?; alpha: 5; nested: {\"k\":[1,\"x\"]}", e.blocks[0].text)
        assertEquals("alpha: second", e.blocks[1].text)
        assertEquals("{\"alpha\": \"second\", \"e\": \"\"}", e.docText.substring(e.blocks[1].charStart, e.blocks[1].charEnd))
    }

    @Test fun jsonObjectWithSingleListMemberIsUnwrapped() {
        val j = "{\"meta\": {\"v\": 1}, \"items\": [{\"a\": \"x\"}, {\"a\": \"y\"}]}"
        val e = run("d.json", j)
        assertEquals(listOf("a: x", "a: y"), e.blocks.map { it.text })
        assertEquals("{\"a\": \"y\"}", e.docText.substring(e.blocks[1].charStart, e.blocks[1].charEnd))
    }

    @Test fun jsonSingleObjectIsOneRecord() {
        val e = run("d.json", "{\"title\": \"T\", \"body\": \"B\\u00e9\\n\"}")
        assertEquals("title: T; body: Bé\n", e.blocks.single().text)
    }

    @Test fun jsonlRecordsAndBadLine() {
        val e = run("d.jsonl", "{\"q\": \"one\"}\n\n{broken\n{\"q\": \"two\", \"n\": 2}\n")
        assertEquals(listOf("q: one", "q: two; n: 2"), e.blocks.map { it.text })
        assertEquals("corrupt_structured", e.issues.single().code)
        assertEquals("warning", e.issues.single().severity)
        assertEquals("{\"q\": \"two\", \"n\": 2}", e.docText.substring(e.blocks[1].charStart, e.blocks[1].charEnd))
    }

    @Test fun malformedJsonIsCorruptStructured() {
        val e = run("d.json", "{\"a\": [1, 2,")
        assertEquals("corrupt_structured", e.issues.single().code)
        assertTrue(e.blocks.isEmpty())
    }

    @Test fun jsonDeepNestingDoesNotOverflow() {
        val e = run("d.json", "[".repeat(100000))
        assertEquals("corrupt_structured", e.issues.single().code)
    }

    @Test fun jsonScalarsOnlyIsEmptyContent() {
        assertEquals("empty_content", run("d.json", "[1, 2, \"x\"]").issues.single().code)
    }
}
