from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional

class Review(TypedDict):
    Product_Name: Annotated[Optional[str], "Name of the product under review"]
    Reviewer_Name: Annotated[Optional[str], "Name of the person who wrote the review"]
    summary: Annotated[str, "Summarise the review in about 25 words"]
    sentiment: Literal["positive", "negative"]
    pros: Annotated[list[str], "List the pros of the product mentioned in the review"]
    pros: Annotated[list[str], "List the cons of the product mentioned in the review"]
    rating: Annotated[int, "Stars given by the reviewer out of 5. Make it out of 5 if it is out of 10"]



load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
I bought the SoundBeat Pro wireless earbuds two weeks ago. The sound quality is fantastic, with deep bass and clear vocals. The battery lasts around 7 hours on a single charge, and the charging case provides two additional charges. They are lightweight and comfortable for short listening sessions. However, the touch controls are frustratingly sensitive, and the microphone sounds muffled during phone calls. The noise cancellation is also weaker than advertised, especially in crowded places. I paid ₹2,499 for them, and considering the sound quality and battery life, I think they offer decent value for money. Overall, I'm satisfied with my purchase, but I wouldn't recommend them to people who frequently take calls.

Reviewer: Rahul Sharma
Product: SoundBeat Pro Wireless Earbuds
Rating: 3/5 stars
"""
)

print(result)