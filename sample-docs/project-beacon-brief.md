# Project Beacon — Brief

## Overview

Project Beacon is the migration of the legacy CRM monolith to microservices on AWS. The target architecture uses Amazon Kinesis for event streaming, DynamoDB for the customer store, and Go services deployed through GitHub Actions.

## Timeline

Phase 1 (Q2 2026): stand up the new contact service and dual-write from the monolith. Phase 2 (Q3 2026): move read traffic to the contact service and start extracting the campaign module. Phase 3 (Q1 2027): retire the monolith's CRM schema.

## Team

The Beacon team is 8 engineers, 2 QA, and a product owner. The principal engineer owns the architecture decision records, which live in the repository under docs/adr rather than the team wiki.

## Success Criteria

Contact lookup p99 under 250 ms, no lost customer records during dual-write verified by nightly reconciliation, and a support-ticket volume no worse than the monolith baseline in the first month.
