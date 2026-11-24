FROM node:22-alpine
WORKDIR /app
COPY dashboard-next/package.json ./
RUN npm install
COPY dashboard-next ./
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
