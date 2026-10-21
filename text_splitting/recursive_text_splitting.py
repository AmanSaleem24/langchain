from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('LostLink_Review_Paper.pdf')

docs = loader.load()

text = """
On the edge of a quiet coastal village stood an old lighthouse that had been abandoned for decades. Its white paint had faded, its windows were cracked, and wild plants had begun growing around its stone walls. Most villagers believed it was haunted, while others thought it was simply a useless reminder of the past.

One rainy evening, a seventeen-year-old boy named Daniel discovered an old notebook in his grandfather's attic. The notebook belonged to his great-grandfather, who had once been the lighthouse keeper. Curious, Daniel opened it and began reading. The pages contained stories about violent storms, missing ships, and a mysterious light that appeared even when the lighthouse lamp was switched off.

One sentence caught his attention: "When the sea grows silent, follow the light that has no flame."

Unable to ignore his curiosity, Daniel decided to visit the lighthouse that night. He packed a flashlight, a jacket, and the notebook before leaving home. The rain had stopped, but thick clouds covered the moon. As he approached the lighthouse, he noticed something strange. A faint blue glow flickered behind one of the broken windows.

Daniel climbed the narrow staircase inside. Every step made a creaking sound, and cold wind rushed through the cracks in the walls. At the top, he found an old brass compass resting beside a dusty lantern. The compass needle was spinning wildly, pointing toward a small wooden box hidden beneath the floorboards.

Inside the box lay a collection of letters and a photograph of a ship that had disappeared many years earlier. The final letter explained that the lighthouse keeper had discovered a dangerous underwater reef that caused ships to crash during storms. He had built a special signaling device to warn sailors, but before he could reveal its location, a terrible storm destroyed the equipment.

Daniel suddenly understood why the blue light appeared. The old signaling device, powered by a forgotten mechanism beneath the lighthouse, was still functioning.

The following morning, Daniel showed the letters to the village council. With the help of local engineers, they repaired the lighthouse and restored its warning system. Months later, when a powerful storm approached the coast, the lighthouse sent out its blue signal for the first time in decades. A fishing boat changed course just in time, avoiding the hidden reef.

From that day onward, the villagers no longer feared the lighthouse. They celebrated it as a symbol of courage, discovery, and the importance of remembering the past. And Daniel learned that sometimes the greatest adventures begin with a simple decision to investigate something everyone else has forgotten.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100, 
    chunk_overlap = 10,
)
result = splitter.split_text(text)

print(len(text))
print(len(text) // 100)

print(len(result))
print(result[0])

print(result)