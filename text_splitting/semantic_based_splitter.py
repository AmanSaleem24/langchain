from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()


loader = PyPDFLoader('LostLink_Review_Paper.pdf')

docs = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.25
)



text = """
The Amazon rainforest is one of the most biodiverse ecosystems on Earth, covering millions of square kilometers across South America. It provides habitat for countless species of plants, insects, birds, and mammals, while its vegetation plays an important role in the global carbon cycle. Meanwhile, the development of quantum computing is transforming how scientists approach complex computational problems. Unlike classical computers, which use bits representing either zero or one, quantum computers use qubits that can exploit superposition and entanglement to perform certain types of calculations more efficiently. Although practical quantum computing still faces major engineering challenges, researchers are exploring its potential applications in cryptography, chemistry, and materials science.

The history of the printing press illustrates how technological innovations can reshape entire societies. Around the middle of the fifteenth century, Johannes Gutenberg developed a movable-type printing system in Europe that made producing books faster and less expensive than copying them by hand. As printed materials became more accessible, knowledge spread more widely, literacy gradually increased, and scientific ideas circulated across geographical boundaries. The printing revolution contributed to major intellectual and cultural changes, including the Renaissance and the Reformation, demonstrating how improvements in communication technology can influence education, politics, and human civilization.
"""


docs = splitter.create_documents([text])
print(len(docs))
print(docs)

