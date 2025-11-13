$accessKey = ""
$secretKey = ""
$bucketName = "itesm-mna/202502-equipo1/"
$region = "us-east-2" 

aws configure --profile equipo1

# Configurar AWS CLI
aws configure set aws_access_key_id $accessKey
aws configure set aws_secret_access_key $secretKey
aws configure set default.region $region
aws configure set default.output json


# Inicializar DVC si no está inicializado
if (!(Test-Path ".dvc")) {
    dvc init
}

# Configurar almacenamiento remoto de DVC
dvc remote add -d s3remote "s3://$bucketName"
dvc remote modify s3remote access_key_id $accessKey
dvc remote modify s3remote secret_access_key $secretKey
dvc remote modify s3remote region $region

Write-Host "✅ DVC configurado con AWS S3 correctamente."
