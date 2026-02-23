def tool_execution_loop(llm, messages, tools, max_loops=5):
    loop_count = 0

    while loop_count < max_loops:
        response = llm.create_chat_completion(
            messages=messages,
            tools=tools,
            stream=False
        )

        message = response["choices"][0]["message"]
        content = message.get("content", [])

        tool_calls = [b for b in content if b.get("type") == "tool_use"]

        if not tool_calls:
            return response

        from .executor import execute_parallel
        results = execute_parallel(tool_calls)

        for call, result in zip(tool_calls, results):
            messages.append({
                "role": "assistant",
                "content": [call]
            })

            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": str(result)
                }]
            })

        loop_count += 1

    return response
