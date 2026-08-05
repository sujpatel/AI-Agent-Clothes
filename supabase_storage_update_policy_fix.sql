-- Run this now. upload_photo() uses upsert=true, which Supabase treats as
-- a potential UPDATE — without an UPDATE policy, every upload fails with
-- "new row violates row-level security policy" even with a fully correct
-- token and path, because Storage's internal upsert check requires it.

create policy "Users can update their own photos"
    on storage.objects for update
    using (bucket_id = 'photos' and (storage.foldername(name))[1] = auth.uid()::text)
    with check (bucket_id = 'photos' and (storage.foldername(name))[1] = auth.uid()::text);
