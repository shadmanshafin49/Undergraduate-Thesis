# Chaincode-as-a-Service image for the DB-BOA chaincode.
# Built by the HOST docker (works on Docker 29); run as a container that the
# Fabric peers connect to over gRPC — avoids the peer-side chaincode image
# build that is incompatible with Docker 29 + Fabric 2.5.
FROM node:18-slim

ARG CC_SERVER_PORT=9999
WORKDIR /usr/src/app

COPY package.json ./
RUN npm install --omit=dev && npm cache clean --force

COPY . .

ENV CHAINCODE_SERVER_ADDRESS=0.0.0.0:${CC_SERVER_PORT}
EXPOSE ${CC_SERVER_PORT}

CMD ["npm", "run", "start:server"]
