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
