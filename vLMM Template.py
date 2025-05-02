# Databricks notebook source
PIP_REQUIREMENTS = (
    "openai vllm==0.6.4.post1 httpx==0.27.2 "
    "transformers==4.46.3 accelerate==1.0.0 "
    "mlflow==2.19.0 "
    "git+https://github.com/stikkireddy/mlflow-extensions.git@v0.17.0"
)
%pip install {PIP_REQUIREMENTS}

dbutils.library.restartPython()

# COMMAND ----------

PIP_REQUIREMENTS = (
    "openai==1.45.0 vllm==0.6.4.post1 httpx==0.27.2 "
    "transformers==4.46.3 accelerate==1.0.0 "
    "mlflow==2.19.0 "
    "git+https://github.com/stikkireddy/mlflow-extensions.git@v0.17.0"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Change the below configs

# COMMAND ----------

CATALOG = 'mlops_pj'
SCHEMA = 'gsk_gsc_cfu_count'
MODEL_NAME = 'internvl-72b-awq'
ENDPOINT_NAME = 'internvl-72b_awq_instruct'
# LOCAL_PATH_TO_MODEL = '/Volumes/mlops_pj/gsk_gsc_cfu_count/hf_model/Qwen2-VL-2B-Instruct/'

# COMMAND ----------

from mlflow_extensions.serving.engines import VLLMEngineProcess
from mlflow_extensions.serving.engines.vllm_engine import VLLMEngineConfig
from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig, ServingConfig,EzDeployVllmOpenCompat

# COMMAND ----------

deployer = EzDeployVllmOpenCompat(
  config= EzDeployConfig(
    name="InternVL2_5-78B-MPO-AWQ",
    engine_proc=VLLMEngineProcess,
    engine_config=VLLMEngineConfig(
          model="OpenGVLab/InternVL2_5-78B-MPO-AWQ", # copy the Hf link
          guided_decoding_backend="outlines",
          vllm_command_flags={
              "--gpu-memory-utilization": 0.97,
              "--quantization" : "awq",
              "--dtype": "float16",
              "--enforce-eager": None,
              "--enable-auto-tool-choice" : None,
              "--tool-call-parser" : "hermes"
          },
          max_model_len=15000,
          max_num_images=2
),
  serving_config=ServingConfig(
      # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
      minimum_memory_in_gb=50,
  ),
  pip_config_override = PIP_REQUIREMENTS.split(" ")
),
  registered_model_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
)

# COMMAND ----------

deployer.download()

# COMMAND ----------

deployer.register()

# COMMAND ----------

# MAGIC %md
# MAGIC # Below is the code to deploy the endpoint to model serving

# COMMAND ----------

# deployer.deploy(ENDPOINT_NAME, scale_to_zero=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kill the existing process and reload model from the UC (Run the below cell every time you want to restart the process)

# COMMAND ----------

from mlflow_extensions.testing.helper import kill_processes_containing

kill_processes_containing("vllm")
kill_processes_containing("ray")
kill_processes_containing("from multiprocessing")

# COMMAND ----------

# from mlflow.models import validate_serving_input
import mlflow
model_uri =f"models:/{CATALOG}.{SCHEMA}.{MODEL_NAME}/8"

pyfunc_model = mlflow.pyfunc.load_model(model_uri)

base_url = str(pyfunc_model.unwrap_python_model()._engine._server_http_client.base_url)

print("base_url:",base_url)


# COMMAND ----------

# MAGIC %md
# MAGIC # Using the predict function to prompt the llm

# COMMAND ----------

# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'
serving_payload = {
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "temperature": 1.0,
  "max_tokens": 10,

}
pyfunc_model.predict(serving_payload)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Test Prompting on an Image
# MAGIC
# MAGIC ## Ask a question to an image

# COMMAND ----------

import urllib.request
from PIL import Image
from io import BytesIO 

image_url = "https://www.arsenal.com/sites/default/files/styles/large_16x9/public/images/saka-celeb-bayern.png?h=3c8f2bed&auto=webp&itok=Twjeu8tug"
with urllib.request.urlopen(image_url) as url:
    img = Image.open(BytesIO(url.read()))

display(img)

# COMMAND ----------

from openai import OpenAI
client = OpenAI(
  base_url=f"{base_url}/v1",
  api_key="DUMMY"
)

# COMMAND ----------

from openai import OpenAI
client = OpenAI(
  base_url=f"{base_url}/v1",
  api_key="DUMMY"
)

response = client.chat.completions.create(
  model="default",
  messages=[
    {"role": "user", 
    "content": [
                {"type": "text", "text": "which football team is this player belong to?"},
                {
                    "type": "image_url",
                    "image_url": {
                      "url": image_url
                    },
                },
            ],
    }
  ],
  temperature=0.0,
  max_tokens=150,

)

response.choices[0].message.content.strip()

# COMMAND ----------

# MAGIC %md
# MAGIC # Test Prompting on an Image
# MAGIC # Using tool calling to design an agentic flow

# COMMAND ----------


response = client.chat.completions.create(
  model="default",
  messages=[
    {"role": "user", 
    "content": [
                 {"type": "text", "text": "which football team is this player belong to use tools?"},
                {
                    "type": "image_url",
                    "image_url": {
                      "url": image_url
                    },
                },
            ],
    }
  ],
  tools = [{
    "type": "function",
    "function": {
        "name": "get_player_information",
        "description": "This functions gets all details about a player shown in the image including where he is male/female and club affiliation",
        "parameters": {
            "type": "object",
            "properties": {
                "team": {
                    "type":
                    "string",
                    "description":
                    "The sports club or team the player belongs to can be from any sport"
                },
                "gender": {
                    "type":
                    "string",
                    "description":
                    "The gender of the player can be either male or female or 'no player found'",
                },
            },
            "required": ["team", "gender"]
        }
    }
}],
tool_choice={"type": "function", "function": {"name": "get_player_information"}},
  temperature=0.0,
  max_tokens=150,

)

response.choices[0].message.content.strip()

# COMMAND ----------

response

# COMMAND ----------

# MAGIC %md
# MAGIC #Load files from Volumes

# COMMAND ----------

df_raw = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobfilter", f"*.jpg")
    .load(f"/Volumes/{CATALOG}/{SCHEMA}/jd_images")
)

# COMMAND ----------

df_raw.writeStream.trigger(availableNow=True).option(
        "checkpointLocation",
        f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/raw_imgs",
).toTable(f"{CATALOG}.{SCHEMA}.raw_img_bytes").awaitTermination()

# COMMAND ----------

df_img = spark.table(f"{CATALOG}.{SCHEMA}.raw_img_bytes")
display(df_img)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, regexp_replace
import pandas as pd


@pandas_udf("string")
def classify_img(images: pd.Series) -> pd.Series:

    from io import BytesIO 
    import base64
    from openai import OpenAI

    def classify_one_image(img): # We could update this to tak multiple parameters
        client = OpenAI(
          base_url=f"{base_url}/v1",
          api_key="DUMMY")

        image_file = BytesIO(img)
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user", 
                    "content": [
                            {
                                 "type": "text", 
                                 #"text": "This images contains a human, your task is to tell me if the person is a male or female, make sure to #answer with only the letter M for male and F for female "
                                  "text": "This images contains a human, your task is to tell me what this person is doing and try to identify who they are "
                            },
                            {
                                "type": "image_url",
                                "image_url": { "url": f"data:image/png;base64,{image_base64}"} 
                            }
                    ]
                }
            ]
        )

        return response.choices[0].message.content.strip()
    return pd.Series([classify_one_image(img) for img in images])

# COMMAND ----------

df_inference = df_img.repartition(4).withColumn("vLLM_predict", classify_img("content"))

# COMMAND ----------

display(df_inference)

# COMMAND ----------

from mlflow_extensions.testing.helper import kill_processes_containing

kill_processes_containing("vllm")
kill_processes_containing("ray")
kill_processes_containing("from multiprocessing")

# COMMAND ----------


