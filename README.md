# GET IN – AI Reservation Assistant (Discovery Phase)

Este repositório contém o código desenvolvido durante a **Discovery Técnica** do projeto **GET IN – Assistente Conversacional de Reservas**.

O artefato principal do projeto é o relatório **“Discovery Técnica – GetIn”**, que consolida arquitetura, resultados das PoCs, riscos e análise de custos.

---

## Objetivo

Validar a viabilidade técnica de um assistente conversacional capaz de:

- Interpretar linguagem natural em português
- Consultar conhecimento interno do restaurante (RAG)
- Executar operações reais na API GET IN (reservas, consultas, cancelamentos)
- Operar de forma controlada em fluxos multi-turno.

---

## Estrutura do Projeto

```text
pocs/
  poc1_api_getin/        # Integração direta com a API GET IN
  poc2_rag/              # Camada de conhecimento (RAG)
  poc3_agent/            # Agente conversacional e orquestração
  poc4_multi-turn/       # Fluxo de reserva fim a fim

src/
  agent/                 # Núcleo do agente conversacional
  rag/                   # Pipeline RAG desacoplado
  services/              # Clientes de APIs externas (GET IN)
  core/                  # Configuração
  utils/                 # Utilitários (CLI, UI)
```

## Como rodar o projeto

### Pré-requisitos

- **Python 3.10+** (recomendado Python 3.11)
- `pip`
- Acesso a uma **API key do GET IN (sandbox)**

---

### 1. Criar e ativar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2. Configurar variáveis de ambiente

```bash
GETIN_API_BASE_URL=https://sandbox.getinapis.com/apis/v2
GETIN_API_KEY=SEU_API_KEY_AQUI
GETIN_DEFAULT_UNIT_ID=ID_DA_UNIDADE
```
### 3. Executar as Provas de Conceito

```bash
python -m pocs.poc1_api_getin.run_poc
python -m pocs.poc2_rag.run_poc
python -m pocs.poc3_agent.run_poc
python -m pocs.poc4_multi-turn.run_poc
```