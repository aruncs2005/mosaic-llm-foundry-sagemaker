region=us-west-2
dlc_account_id=763104351884
image=mosaic-llm-foundry-dlc
tag=latest

aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $dlc_account_id.dkr.ecr.$region.amazonaws.com
chmod +x build_and_push.sh; bash build_and_push.sh $dlc_account_id $region $image $tag 