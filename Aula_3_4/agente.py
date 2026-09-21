
from pathlib import Path
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
 
 
from dotenv import load_dotenv
from openai import OpenAI
 
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)
 
api_key_minha = os.getenv("OPENAI_API_KEY", "").strip()
 
if not api_key_minha:
    raise RuntimeError("OPENAI_API_KEY não foi encontrado no arquivo .env.")
 
if not api_key_minha.startswith("sk-"):
    raise RuntimeError("OPENAI_API_KEY não é válido. Certifique-se de que a chave começa com 'sk-'.")
 
 
class QualProdutoinIndicado(BaseModel):
    modelo_indicado: str = Field("Qual modelo do produto é mais indicado para o ambiente")
    motivo: str = Field("Motivo pelo qual o modelo é mais indicado para o ambiente")
 
parseador = PydanticOutputParser(pydantic_object=QualProdutoinIndicado)
 
prompt_produto = PromptTemplate(
    template= """Crie um agente de suporte a vendas para o produto {produto} O agente
           deve ser capaz de responder a perguntas sobre o produto, destacando o modelo, as características para uso na
           {ambiente}
            O agente deve ser amigável, profissional e persuasivo.
 
            {formato_de_saida}""",
    input_variables=["produto", "ambiente"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()}
)
 
 
 
 
modelo = ChatOpenAI(
    model_name="gpt-4o-mini",
    api_key=api_key_minha,
    temperature=0.5
)
 
 
cadeia_1 = prompt_produto | modelo | parseador
 
indicacao = cadeia_1.invoke({"produto": "TV 55 polegadas", "ambiente": "sala de estar"})
 
print(indicacao)


