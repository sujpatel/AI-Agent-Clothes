-- Run this after creating a private bucket named "photos" in the Supabase
-- dashboard (Storage > New bucket > leave "Public bucket" off).
--
-- Files are stored as "{user_id}/{item_id}.jpg" — these policies restrict
-- every operation to files whose first path segment matches the caller's
-- own auth.uid(), the same scoping already used for local photo folders
-- and the conversations table.

create policy "Users can read their own photos"
    on storage.objects for select
    using (bucket_id = 'photos' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Users can upload their own photos"
    on storage.objects for insert
    with check (bucket_id = 'photos' and (storage.foldername(name))[1] = auth.uid()::text);

-- upload_photo() calls upload with upsert=true (so re-adding the same
-- item_id overwrites cleanly) — Supabase treats an upsert as a potential
-- UPDATE, which needs its own policy separate from INSERT, or the RLS check
-- fails with "new row violates row-level security policy" even though the
-- token and path are both correct.
create policy "Users can update their own photos"
    on storage.objects for update
    using (bucket_id = 'photos' and (storage.foldername(name))[1] = auth.uid()::text)
    with check (bucket_id = 'photos' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Users can delete their own photos"
    on storage.objects for delete
    using (bucket_id = 'photos' and (storage.foldername(name))[1] = auth.uid()::text);
