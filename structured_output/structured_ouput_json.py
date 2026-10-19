from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

{
  "title": "Review",
  "type": "object",
  "properties": {
    "Product_Name": {
      "type": "string",
      "description": "Name of the product under review"
    },
    "Reviewer_Name": {
      "type": "string",
      "description": "Name of the person who wrote the review"
    },
    "summary": {
      "type": "string",
      "description": "Summarise the review in about 25 words"
    },
    "sentiment": {
      "type": "string",
      "enum": ["positive", "negative"],
      "description": "Sentiment of the review"
    },
    "pros": {
      "anyOf": [
        {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        {
          "type": "null"
        }
      ],
      "description": "List the pros of the product mentioned in the review"
    },
    "cons": {
      "anyOf": [
        {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        {
          "type": "null"
        }
      ],
      "description": "List the cons of the product mentioned in the review"
    },
    "rating": {
      "type": "integer",
      "description": "Stars given by the reviewer out of 5. Make it out of 5 if it is out of 10"
    }
  },
  "required": [
    "Product_Name",
    "Reviewer_Name",
    "summary",
    "sentiment",
    "pros",
    "cons",
    "rating"
  ]
}



load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")
review_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "Product_Name": {
      "type": "string",
      "description": "Name of the product under review"
    },
    "Reviewer_Name": {
      "type": "string",
      "description": "Name of the person who wrote the review"
    },
    "summary": {
      "type": "string",
      "description": "Summarise the review in about 25 words"
    },
    "sentiment": {
      "type": "string",
      "enum": ["positive", "negative"],
      "description": "Sentiment of the review"
    },
    "pros": {
      "anyOf": [
        {"type": "array", "items": {"type": "string"}},
        {"type": "null"}
      ],
      "description": "List the pros of the product mentioned in the review"
    },
    "cons": {
      "anyOf": [
        {"type": "array", "items": {"type": "string"}},
        {"type": "null"}
      ],
      "description": "List the cons of the product mentioned in the review"
    },
    "rating": {
      "type": "integer",
      "description": "Stars given by the reviewer out of 5. Make it out of 5 if it is out of 10"
    }
  },
  "required": [
    "Product_Name",
    "Reviewer_Name",
    "summary",
    "sentiment",
    "pros",
    "cons",
    "rating"
  ]
}
structured_model = model.with_structured_output(review_schema)

result = structured_model.invoke("""
I bought the SoundBeat Pro wireless earbuds two weeks ago. The sound quality is fantastic, with deep bass and clear vocals. The battery lasts around 7 hours on a single charge, and the charging case provides two additional charges. They are lightweight and comfortable for short listening sessions. However, the touch controls are frustratingly sensitive, and the microphone sounds muffled during phone calls. The noise cancellation is also weaker than advertised, especially in crowded places. I paid ₹2,499 for them, and considering the sound quality and battery life, I think they offer decent value for money. Overall, I'm satisfied with my purchase, but I wouldn't recommend them to people who frequently take calls.

Reviewer: Rahul Sharma
Product: SoundBeat Pro Wireless Earbuds
Rating: 3/5 stars
"""
)

print(result)
print(type(result))