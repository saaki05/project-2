FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY predictiq-data-pipeline /src/predictiq-data-pipeline
COPY predictiq-research /src/predictiq-research
COPY predictiq-deployment /src/predictiq-deployment
RUN pip install --no-cache-dir /src/predictiq-data-pipeline /src/predictiq-research \
    /src/predictiq-deployment
EXPOSE 8000
ENTRYPOINT ["predictiq-serve", "--host", "0.0.0.0", "--bundle", "/models/bundle.pt"]
