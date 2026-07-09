# Project Atlas — Brief

## Overview

Project Atlas is the migration of the legacy billing monolith to event-driven microservices on Azure. The target architecture uses Azure Service Bus for messaging, Cosmos DB for the transaction ledger, and ASP.NET Core services deployed via Azure DevOps pipelines.

## Timeline

Phase 1 (Q1 2026): extract the invoicing module and run it in shadow mode alongside the monolith. Phase 2 (Q2 2026): cut over invoicing traffic and begin extracting the payments module. Phase 3 (Q3–Q4 2026): decommission the monolith's billing tables.

## Team

The Atlas team consists of 6 engineers, 1 QA, and a delivery manager. The tech lead is responsible for the architecture decision records (ADRs), which are stored in the team wiki under /atlas/adr.

## Success Criteria

Billing latency p99 under 400 ms, zero data loss during cutover verified by ledger reconciliation, and infrastructure cost reduction of at least 15% compared to the monolith baseline.
