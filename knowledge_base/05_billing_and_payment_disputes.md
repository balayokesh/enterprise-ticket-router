# Billing, Payment Processing & Invoice Support SOP

## Section 1: Failed Payments & Card Declines
When a customer encounters a payment issue during renewal or checkout, advise verifying the billing ZIP/postal code, card expiration date, and international transaction permissions with their card issuer before retrying payment in the Billing Portal.

## Section 2: Duplicate Charges & Unrecognized Invoices
If a customer reports a duplicate charge or unexpected invoice for software or hardware orders, cross-check the Transaction ID and order history. If a duplicate authorization settled, initiate a full refund for the duplicate entry and send an updated itemized receipt.

## Section 3: Escalation Criteria for Unresolved Billing Issues
Any billing or account dispute where the customer has contacted support multiple times without resolution, or where a cancellation request failed to stop a charge, must be flagged `AI_Escalate = True` and routed to a Senior Billing Specialist within 4 business hours.
