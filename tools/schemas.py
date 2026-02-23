import jsonschema

def validate_tool_input(schema, input_data):
    jsonschema.validate(instance=input_data, schema=schema)
