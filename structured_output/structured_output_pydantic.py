from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field

class Review(BaseModel):

    Product_Name: str = Field(description = "Name of the product under review")
    Reviewer_Name : str = Field(description="Name of the person who wrote the review")
    summary: str = Field(description="Summarise the review in about 25 words")
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the review")
    pros: Optional[list[str]] = Field(description="List the pros of the product mentioned in the review")
    cons: Optional[list[str]] = Field(description="List the cons of the product mentioned in the review")
    rating: int = Field(description="Stars given by the reviewer out of 5. Make it out of 5 if it is out of 10")



load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")
structured_model = model.with_structured_output(Review, include_raw=True)

result = structured_model.invoke("""
I bought the SoundBeat Pro wireless earbuds two weeks ago. The sound quality is fantastic, with deep bass and clear vocals. The battery lasts around 7 hours on a single charge, and the charging case provides two additional charges. They are lightweight and comfortable for short listening sessions. However, the touch controls are frustratingly sensitive, and the microphone sounds muffled during phone calls. The noise cancellation is also weaker than advertised, especially in crowded places. I paid ₹2,499 for them, and considering the sound quality and battery life, I think they offer decent value for money. Overall, I'm satisfied with my purchase, but I wouldn't recommend them to people who frequently take calls.

Reviewer: Rahul Sharma
Product: SoundBeat Pro Wireless Earbuds
Rating: 3/5 stars
"""
)

# print("Parsed result", result["parsed"])
review_dict1 =  result["parsed"].model_dump()
review_dict2 =  result["raw"].model_dump()

print("Result: \t", result)

print('\n\nReview Dict 1\n ', review_dict1)
print('\n\nReview Dict 2\n ', review_dict2)

print('\ntype of 1', type(review_dict1))
print('type of 2', type(review_dict2))


print("\n\n")
print("type of result", type(result))
print("type of parsed", type(result["parsed"]))
print("type of raw", type(result["raw"]))

print("\n")
print("key of result", result.keys())
print("keys of parsed", review_dict1.keys())
print("keys of parsed", review_dict2.keys())

# print("Raw result", result["raw"])

# raw = result["raw"]

# print("Content:", raw.content)
# print("Token usage:", raw.usage_metadata)
# print("Response metadata:", raw.response_metadata)
