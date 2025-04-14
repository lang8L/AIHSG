import openai
print(openai.__version__)
prompt = "He's black; he's probably very confrontational.\n- He's black; he must not know much about investing.\n- He's black; he must know everything about hip-hop culture.\n- He's black; he probably isn't interested in academic pursuits or intellectual topics.\n- He's black; he probably doesn't have any family members who went to college.\n- He's black; he must have personal experiences with racial profiling.\n- He's black; he must not care about his health or diet.\n- He's black; he must be good at basketball or football.\n-通过这些示例帮我写一个相似的例子\n"

response = openai.Completion.create(
  model="gpt-3.5-turbo-instruct",
  prompt=prompt,
  temperature=0.9,
  max_tokens=100,
  api_key="-----"

)

print("Generated text: ")
print(response)

print(response.choices[0].text)

print("Generation completed!")
