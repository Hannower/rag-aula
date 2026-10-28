FROM node:24-alpine

WORKDIR /app

# Instala as dependências da interface no volume do container.
COPY package.json package-lock.json ./
RUN npm ci

EXPOSE 3000

CMD ["npm", "run", "dev"]
