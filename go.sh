creds_path="/package/MNIST/configs/Google_credentials.json"
full_path="${PWD}${creds_path}"
export GOOGLE_APPLICATION_CREDENTIALS=$full_path
#dvc pull
dvc repro eval.dvc
