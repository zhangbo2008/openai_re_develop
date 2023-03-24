#=https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb   依赖这套代码开发一个openai自定义模型加强回复效果.


import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken

#=两个初始model的名字.
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key = "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"
prompt = "Who won the 2020 Summer Olympics men's high jump?"
a1=openai.Completion.create(
    prompt=prompt,
    temperature=0,
    max_tokens=300,
    model=COMPLETIONS_MODEL
)
a=openai.Completion.create(
    prompt=prompt,
    temperature=0,
    max_tokens=300,
    model=COMPLETIONS_MODEL
)["choices"][0]["text"].strip(" \n")
print(a) #=======这个回答是错的.
print(1)







prompt = """Answer the question as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know".

Q: Who won the 2020 Summer Olympics men's high jump?
A:"""

print(openai.Completion.create(
    prompt=prompt,
    temperature=0,
    max_tokens=300,
    model=COMPLETIONS_MODEL
)["choices"][0]["text"].strip(" \n"))







#===============加入context
prompt = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know"

Context:
The men's high jump event at the 2020 Summer Olympics took place between 30 July and 1 August 2021 at the Olympic Stadium.
33 athletes from 24 nations competed; the total possible number depended on how many nations would use universality places 
to enter athletes in addition to the 32 qualifying through mark or ranking (no universality places were used in 2021).
Italian athlete Gianmarco Tamberi along with Qatari athlete Mutaz Essa Barshim emerged as joint winners of the event following
a tie between both of them as they cleared 2.37m. Both Tamberi and Barshim agreed to share the gold medal in a rare instance
where the athletes of different nations had agreed to share the same medal in the history of Olympics. 
Barshim in particular was heard to ask a competition official "Can we have two golds?" in response to being offered a 
'jump off'. Maksim Nedasekau of Belarus took bronze. The medals were the first ever in the men's high jump for Italy and 
Belarus, the first gold in the men's high jump for Italy and Qatar, and the third consecutive medal in the men's high jump
for Qatar (all by Barshim). Barshim became only the second man to earn three medals in high jump, joining Patrik Sjöberg
of Sweden (1984 to 1992).

Q: Who won the 2020 Summer Olympics men's high jump?
A:"""

print(openai.Completion.create(
    prompt=prompt,
    temperature=0,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    model=COMPLETIONS_MODEL
)["choices"][0]["text"].strip(" \n"))



# Adding extra information into the prompt only works when the dataset of extra content that the model may need to know is small enough to fit in a single prompt. What do we do when we need the model to choose relevant contextual information from within a large body of information?










# We have hosted the processed dataset, so you can download it directly without having to recreate it.
# This dataset has already been split into sections, one row for each section of the Wikipedia page.

# df = pd.read_csv('https://cdn.openai.com/API/examples/data/olympics_sections_text.csv')
df=pd.DataFrame({'学科':['数学','物理学','化学','生物'],
                '去向':['（1）数学国家队共30人，去北京大学19人，清华大学9人，麻省理工学院2人。',
   '物理学国家队共25人，去北京大学24人，清华大学1人，麻省理工学院0人。',
   '（3）化学国家队共20人，去北京大学16人，清华大学4人，麻省理工学院0人。',
 '（4）生物学国家队共20人，去北京大学7人，清华大学13人，麻省理工学院0人。',]
 }
    )
# df = df.set_index(["title", "heading"])
print(f"{len(df)} rows in the data.")
# df.sample(5)
print(11111111111)



def get_embedding(text: str, model: str=EMBEDDING_MODEL) :
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) :
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r['去向']) for idx, r in df.iterrows()
    }


def load_embeddings(fname: str) :
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    } # r是行的意思. 然后取出他的嵌入向量.

# document_embeddings = load_embeddings("https://cdn.openai.com/API/examples/data/olympics_sections_document_embeddings.csv")
document_embeddings = compute_doc_embeddings(df)

#===查看一下.
example_entry = list(document_embeddings.items())[0] # items. 返回二重列表.
print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")




def vector_similarity(x, y) :
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts) :
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities





# order_document_sections_by_query_similarity("Who won the men's high jump?", document_embeddings)[:5]


# order_document_sections_by_query_similarity("Who won the women's high jump?", document_embeddings)[:5]



MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR)) # 各个部分需要分隔符.

f"Context separator contains {separator_len} tokens"

import tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\\n{content}<im_end>\\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See <https://github.com/openai/openai-python/blob/main/chatml.md> for information on how messages are converted to tokens.""")


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
#========下面的逻辑是我们尽量给question,匹配相关上下文,直到扩充到500.MAX_SECTION_LEN目前是500,可以根据自己问题复杂性进行修改.
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += num_tokens_from_string(document_section['去向'],EMBEDDING_MODEL) + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section['去向'].replace("\n", " ")) # 文档的每一个部分需要分隔符来进行分开. 可能为了跟chatgpt的训练数据保持格式一致!!!!!不要自己修改格式.
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:" #下面进行拼接,


# prompt = construct_prompt(
#     "Who won the 2020 Summer Olympics men's high jump?",
#     document_embeddings,
#     df
# )

# print("===\n", prompt)





COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}
def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings,
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")
#================核心用例入口!!!!!!!!!!!!!!



a=answer_query_with_context("物理竞赛的队员去清华和北大一共几个人?", df, document_embeddings)
print(a)

#More Examples

if 0:



    query = "Why was the 2020 Summer Olympics originally postponed?"
    answer = answer_query_with_context(query, df, document_embeddings)

    print(f"\nQ: {query}\nA: {answer}")



    query = "In the 2020 Summer Olympics, how many gold medals did the country which won the most medals win?"
    answer = answer_query_with_context(query, df, document_embeddings)

    print(f"\nQ: {query}\nA: {answer}")


    query = "What was unusual about the men’s shotput competition?"
    answer = answer_query_with_context(query, df, document_embeddings)

    print(f"\nQ: {query}\nA: {answer}")



    query = "In the 2020 Summer Olympics, how many silver medals did Italy win?"
    answer = answer_query_with_context(query, df, document_embeddings)

    print(f"\nQ: {query}\nA: {answer}")


    query = "What is the total number of medals won by France, multiplied by the number of Taekwondo medals given out to all countries?"
    answer = answer_query_with_context(query, df, document_embeddings)

    print(f"\nQ: {query}\nA: {answer}")


    query = "What is the tallest mountain in the world?"
    answer = answer_query_with_context(query, df, document_embeddings)

    print(f"\nQ: {query}\nA: {answer}")



    query = "Who won the grimblesplatch competition at the 2020 Summer Olympic games?"
    answer = answer_query_with_context(query, df, document_embeddings)

    print(f"\nQ: {query}\nA: {answer}")