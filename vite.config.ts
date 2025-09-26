import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./client/src"),
      "@shared": path.resolve(__dirname, "./shared"),
    },
  },
  optimizeDeps: {
    exclude: ["@tensorflow/tfjs"],
  },
  build: {
    commonjsOptions: {
      exclude: ["@tensorflow/tfjs"],
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5000,
  },
});