import json
L=lambda a:[json.loads(l) for l in open(f'{a}/{a}.jsonl')]
A,B=L('AMOFF'),L('AMON')
k=lambda r:(r.get('content'),json.dumps(r.get('tool_calls')),r.get('completion_tokens'),r.get('finish'))
wp=[(a['id'][-12:],a['prompt_tokens']) for a,b in zip(A,B) if a['prompt_tokens']<=16320]
print('within-pool n',len(wp),'diffs',[x for a,b in zip(A,B) for x in [(a['id'][-12:],a['prompt_tokens'],a['completion_tokens'],b['completion_tokens'])] if a['prompt_tokens']<=16320 and k(a)!=k(b)])
print('past-pool diffs',[(a['prompt_tokens'],a['completion_tokens'],b['completion_tokens']) for a,b in zip(A,B) if a['prompt_tokens']>16320 and k(a)!=k(b)])
