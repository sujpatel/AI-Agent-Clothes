-- Run this now — the conversations table already exists, it's just missing
-- the table-level grant that lets logged-in users actually attempt reads/
-- writes (RLS policies alone don't grant that; see the comment in
-- supabase_conversations_table.sql). This is what's causing every real
-- request to /outfit/today and /outfit/override to fail with a 403.

grant select, insert, update on conversations to authenticated;
