\documentclass[11pt]{article}

\usepackage[a4paper,margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{enumitem}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue
}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10}
}

\title{\textbf{House Price Prediction API}}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Overview}
This project is a production-ready machine learning inference API for predicting house prices.
It uses a trained XGBoost regression model and is deployed using FastAPI.
The system includes strict input validation, a custom preprocessing pipeline, and Docker-based containerization.

\section*{Tech Stack}
\begin{itemize}[noitemsep]
    \item Python
    \item FastAPI
    \item Pydantic
    \item XGBoost
    \item Docker
\end{itemize}

\section*{Project Structure}
\begin{lstlisting}
house-price-prediction/
│
├── src/
│   ├── api.py
│   ├── model.py
│   ├── schemas.py
│   ├── preprocessing.py
│
├── models/
│   ├── model.pkl
│   └── column.json
│
├── requirements.txt
├── Dockerfile
└── README.tex
\end{lstlisting}

\section*{ML Workflow}

\subsection*{Training Phase}
\begin{itemize}
    \item Data cleaning and preprocessing
    \item Feature engineering
    \item Model training using XGBoost
    \item Saving trained model and feature metadata
\end{itemize}

Artifacts generated:
\begin{itemize}
    \item \texttt{model.pkl}
    \item \texttt{column.json}
\end{itemize}

\subsection*{Inference Phase}
\begin{enumerate}
    \item Client sends JSON input
    \item FastAPI parses input into a Pydantic model
    \item Input is validated automatically
    \item Data is preprocessed using training metadata
    \item Model predicts house price
    \item Prediction is returned as JSON
\end{enumerate}

\section*{API Endpoints}

\subsection*{/health}
\textbf{Method:} GET

\textbf{Response:}
\begin{lstlisting}
{
  "status": "ok"
}
\end{lstlisting}

\subsection*{/locations}
\textbf{Method:} GET

Returns supported locations loaded from training metadata.
Locations are not derived from user input.

\subsection*{/predict}
\textbf{Method:} POST

\textbf{Request Body:}
\begin{lstlisting}
{
  "location": "Whitefield",
  "total_sqft": 1200,
  "bath": 2,
  "bhk": 3
}
\end{lstlisting}

\textbf{Response:}
\begin{lstlisting}
{
  "predicted_price": 85.23
}
\end{lstlisting}

\section*{Input Validation}
Pydantic models are used to:
\begin{itemize}
    \item Enforce data types
    \item Validate constraints
    \item Reject malformed requests automatically
\end{itemize}

Users send raw JSON; FastAPI internally converts it into Python objects.

\section*{Docker Support}
The project is fully containerized.

\begin{lstlisting}
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
\end{lstlisting}

\section*{Key Design Principles}
\begin{itemize}
    \item Fixed feature space for stable inference
    \item Clear separation of concerns
    \item Production-oriented ML deployment
\end{itemize}

\end{document}
