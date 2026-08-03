-- If storage.objects also turns out to be missing the authenticated role's
-- table-level grant (same root cause as the conversations table fix
-- earlier), this restores it. Supabase normally pre-configures this when
-- you create a bucket through the dashboard, but given we've already found
-- one wrong assumption tonight, this rules the same class of bug out here.
grant select, insert, update, delete on storage.objects to authenticated;
