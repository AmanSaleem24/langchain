import json
from langchain_core.messages import AIMessage
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


result =    {'raw': AIMessage(content=[{'type': 'text', 'text': '{\n  "Product_Name": "SoundBeat Pro Wireless Earbuds",\n  "Reviewer_Name": "Rahul Sharma",\n  "summary": "Rahul Sharma finds theSoundBeat Pro wireless earbuds offer great sound and battery life for the price, but dislikes the sensitive touch controls and poor microphone quality.",\n  "sentiment": "positive",\n  "pros": [\n "Fantastic sound quality with deep bass and clear vocals",\n    "Good battery life of 7 hours plus extra charges from the case",\n    "Lightweight and comfortable design",\n    "Decent value for money"\n  ],\n  "cons": [\n    "Frustratingly sensitive touch controls",\n    "Muffled microphone during phone calls",\n    "Weaker noise cancellation than advertised"\n  ],\n  "rating": 3\n}', 'extras': {'signature': 'EmAKXgFpFH0TTT01t57VLtpndSysKdVPgP3/5hpuQ3zjFyUZqRi8W2lYoTcRUQxMmllbxIaB6rGORunZorD/Woyn7Ljh6AiI6s+NHnzc2ZWA4YFUpZ5j0QyWq/L+4tVAdZM='}}], additional_kwargs={}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3.5-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--01a15578-eeb2-79d3-acf3-553254fa7bc6-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 168, 'output_tokens': 178, 'total_tokens': 346, 'input_token_details': {'cache_read': 0}}), 'parsed': Review(Product_Name='SoundBeat Pro Wireless Earbuds', Reviewer_Name='Rahul Sharma', summary='Rahul Sharma finds the SoundBeat Pro wireless earbuds offer great sound and battery life for the price, but dislikes the sensitive touch controls and poor microphone quality.', sentiment='positive', pros=['Fantastic sound quality with deep bass and clear vocals', 'Good battery life of 7 hours plus extra charges from the case', 'Lightweight and comfortable design', 'Decent value for money'], cons=['Frustratingly sensitive touch controls', 'Muffled microphone during phone calls', 'Weaker noise cancellation than advertised'], rating=3), 'parsing_error': None}






result_dict = {
    "parsed": result["parsed"].model_dump()
    if hasattr(result["parsed"], "model_dump")
    else result["parsed"],
    "raw": result["raw"].model_dump()
    if hasattr(result["raw"], "model_dump")
    else result["raw"],
    "parsing_error": str(result["parsing_error"])
    if result["parsing_error"] else None
}

with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)