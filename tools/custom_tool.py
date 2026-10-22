from langchain_core.tools import tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a*b

result = multiply.invoke({"a":3, "b":4})

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(result)

