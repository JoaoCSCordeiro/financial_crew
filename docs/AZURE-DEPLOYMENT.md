# Deployment Azure — classificador de sentimento fine-tuned

**Estado: documentado, não executado.** Decisão explícita (ver
DECISOES-FINETUNING.md): este projeto não tem uma subscrição Azure ativa
associada, e criar uma só para testar este deployment de ponta a ponta
introduzia custo e complexidade de gestão de credenciais fora do âmbito
"zero-cost, local-first" do resto do Financial Crew. Este documento
descreve os passos exatos e as decisões de arquitetura para quem tiver
(ou quiser criar) uma subscrição Azure e executar isto.

## Escolha: Azure Container Apps (não Azure ML Managed Endpoint)

| | Azure Container Apps | Azure ML Managed Endpoint |
|---|---|---|
| Unidade de deployment | Container Docker genérico | Container + metadados específicos do Azure ML (modelo registado, environment) |
| Free tier | 180.000 vCPU-segundos + 360.000 GiB-segundos + 2M pedidos/mês, **por subscrição, todos os meses** (confirmado em https://azure.microsoft.com/pricing/details/container-apps/, 2026-08-30) | Sem free tier real para inferência em tempo real -- o nó de compute mínimo (`Standard_DS3_v2` ou equivalente) fica ligado 24/7 mesmo sem tráfego |
| Scale-to-zero | Sim (consumption plan) | Não, para real-time endpoints (só batch endpoints escalam a zero) |
| Ajuste ao que já temos | Direto -- já temos um Dockerfile e uma imagem a funcionar (`finetuning/serving/Dockerfile`, testada localmente, ver RESULTS.md) | Exigiria reescrever o serving no formato de scoring script do Azure ML, mais infraestrutura (workspace, registo de modelo) para o mesmo resultado |

Container Apps ganha claramente para este caso: um modelo pequeno (4MB de
adapter + base model), tráfego baixo/demo, e o objetivo é mostrar que o
deployment *pode* correr de ponta a ponta sem custo residual permanente
graças ao scale-to-zero. Azure ML Managed Endpoint faz mais sentido
quando já se usa o resto do ecossistema Azure ML (registo de modelos,
pipelines de treino no próprio Azure) -- não é o caso aqui, o treino corre
localmente (ver `finetuning/`).

## Pré-requisitos (não executados nesta sessão)

- Subscrição Azure ativa (o free tier de 12 meses / créditos de conta nova
  cobre isto confortavelmente para uma demo).
- `az` CLI instalado e autenticado (`az login` -- interativo, tem de ser
  feito pelo utilizador, nunca por um agente).
- A imagem `financial-crew-sentiment` já construída localmente (ver
  `finetuning/README.md`, secção 4) e validada (`docker run` + `curl
  /health` + `curl /classify` -- ver RESULTS.md, secção 3, para a validação
  que já foi feita localmente).

## Passos exatos

```bash
# 1. Grupo de recursos e registo de imagens (Azure Container Registry, tier Basic)
az group create --name rg-financial-crew-sentiment --location westeurope

az acr create --resource-group rg-financial-crew-sentiment \
  --name acrfinancialcrewsentiment --sku Basic

az acr login --name acrfinancialcrewsentiment

# 2. Tag e push da imagem já construída localmente
docker tag financial-crew-sentiment \
  acrfinancialcrewsentiment.azurecr.io/financial-crew-sentiment:v1
docker push acrfinancialcrewsentiment.azurecr.io/financial-crew-sentiment:v1

# 3. Ambiente do Container Apps (consumption plan)
az containerapp env create \
  --name env-financial-crew-sentiment \
  --resource-group rg-financial-crew-sentiment \
  --location westeurope

# 4. Deployment do Container App
az containerapp create \
  --name financial-crew-sentiment \
  --resource-group rg-financial-crew-sentiment \
  --environment env-financial-crew-sentiment \
  --image acrfinancialcrewsentiment.azurecr.io/financial-crew-sentiment:v1 \
  --registry-server acrfinancialcrewsentiment.azurecr.io \
  --target-port 8001 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 2 \
  --cpu 0.5 --memory 1.0Gi

# 5. Ligar o News Analyst ao endpoint público resultante
#    (substitui <fqdn> pelo output do passo 4, campo `properties.configuration.ingress.fqdn`)
export SENTIMENT_SERVICE_URL="https://<fqdn>"
export SENTIMENT_PROVIDER=finetuned
```

`--min-replicas 0` é a decisão que faz o custo tender para zero fora de
uso ativo (scale-to-zero) -- ao custo de uma cold start (o container tem
de arrancar e carregar o modelo, medido localmente em ~8-10s no
`docker run` desta sessão, ver RESULTS.md) na primeira chamada depois de
um período de inatividade.

## O que fica por confirmar (custo real)

O free tier mensal (180k vCPU-s / 360k GiB-s / 2M pedidos) está confirmado
na página pública de pricing. O preço por unidade *depois* de esgotar o
free tier não estava disponível como texto estático na página pública no
momento da pesquisa (a Azure serve esses valores via uma calculadora
JavaScript, não em HTML simples) -- não vou inventar um número aqui.
Antes de decidir se isto fica ligado além de uma demo pontual, confirma o
custo por vCPU-segundo/GiB-segundo em
https://azure.microsoft.com/pricing/calculator/ com a região e a
subscrição reais. Para o volume desta demo (tráfego esporádico, container
minúsculo, scale-to-zero), a expectativa razoável é ficar dentro do free
tier mensal -- mas isso é uma expectativa, não uma medição, e por isso não
está no RESULTS.md.

## Alternativa mais simples ainda (não escolhida, mas válida)

Para uma demo pontual em entrevista sem manter nada "ligado", `az
containerapp create` com `--min-replicas 0` seguido de `az containerapp
delete` no fim da demo evita qualquer custo residual -- mais simples do
que gerir um recurso permanente. Documentado aqui como opção, não como
recomendação por omissão, porque um endpoint permanente (mesmo que barato)
é mais representativo de um cenário de produção real para efeitos de
entrevista.
