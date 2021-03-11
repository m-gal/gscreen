"""
    Script to make predictions on MLflow locally deployed model.

    Created on Jan 2021
    @author: mikhail.galkin
"""
#%%
import sys
import requests

sys.path.extend(["..", "../..", "../../.."])
from gscreen.utils import make_example_df


def main(
    port="8080",  # 8080: serve as MLflow Model. 1234: serve from MLflow Regisry
    host="127.0.0.1",
    headers={"Content-Type": "application/json; format=pandas-split"},
):
    request_uri = f"http://{host}:{port}/invocations"
    example_df, _ = make_example_df(num_rows=2)
    example_json = example_df.to_json(orient="split")
    print(f"{request_uri}")
    print(example_df, "\n")
    print(example_json)

    try:
        response = requests.post(url=request_uri, data=example_json, headers=headers)
        print(f"\nModel response is:")
        print(response.content)
        print("!!! DONE !!!")
    except Exception as ex:
        raise (ex)


if __name__ == "__main__":
    main()

#%%
