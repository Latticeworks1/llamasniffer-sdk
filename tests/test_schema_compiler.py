from llamasniffer.schema_compiler import FieldType, SchemaCompiler


def test_compile_schema_with_explicit_definition():
    compiler = SchemaCompiler()
    config = {
        "name": "custom",
        "description": "desc",
        "schema": {
            "question": "text:Question field",
            "score": {"type": "number", "description": "score", "required": False},
        },
    }

    schema = compiler.compile_schema(config)

    assert schema.name == "custom"
    assert len(schema.fields) == 2
    assert "question" in schema.generation_templates
    assert "_master" in schema.generation_templates


def test_builtin_template_customization():
    compiler = SchemaCompiler()
    config = {
        "dataset_type": "qa_pairs",
        "field_customizations": {"answer": {"description": "Custom answer"}},
    }

    schema = compiler.compile_schema(config)

    answer_field = next(f for f in schema.fields if f.name == "answer")
    assert answer_field.description == "Custom answer"


def test_parse_field_string_handles_unknown_type():
    compiler = SchemaCompiler()
    field_type, description = compiler._parse_field_string("unknown:Something")

    assert field_type == FieldType.TEXT
    assert description == "Something"
