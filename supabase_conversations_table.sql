create table conversations (
    user_id uuid primary key,
    contents jsonb not null,
    occasion text not null,
    updated_at timestamptz not null default now()
);

alter table conversations enable row level security;

create policy "Users can only access their own conversation"
    on conversations
    for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

-- RLS policies only restrict WHICH rows a role can touch — they don't grant
-- the underlying table-level privilege to attempt the operation at all.
-- Without this, every request from a logged-in user gets a 403 regardless
-- of the policy above, since the "authenticated" role has no INSERT/SELECT/
-- UPDATE privilege on this table by default.
grant select, insert, update on conversations to authenticated;
