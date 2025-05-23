
from openai import AzureOpenAI

# Parameters
client = AzureOpenAI(
    azure_endpoint="https://hkust.azure-api.net",
    api_version="2023-05-15",
    api_key="your api"
)


# Function
def get_response(message, model="gpt-4o", instruction="", temperature=0):
    print(message, model)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": message}
        ]
    )

    # print(response.usage)
    return response.choices[0].message.content


