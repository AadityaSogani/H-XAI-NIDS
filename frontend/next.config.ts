import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */

  // cacheComponents (Next 16's PPR-style model) conflicts with `export const
  // dynamic` and would cache live detection results — a stale "no threats
  // found" on a security dashboard is the worst possible bug. Every page here
  // renders against the live API, so caching is off.
  cacheComponents: false,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
