"""Flag submission tool."""

from pydantic_ai import RunContext

from backend.deps import SolverDeps
from backend.tools.core import do_submit_flag


async def submit_flag(ctx: RunContext[SolverDeps], flag: str) -> str:
    """Record the flag when you believe it is correct (local verification only).

    Returns CORRECT when accepted, or DRY RUN if --no-submit is set.
    Do NOT submit placeholder flags like CTF{flag} or CTF{placeholder}.
    """
    if ctx.deps.no_submit:
        return f'DRY RUN — would accept "{flag.strip()}" but --no-submit is set.'

    if ctx.deps.submit_fn:
        display, is_confirmed = await ctx.deps.submit_fn(flag)
    else:
        display, is_confirmed = await do_submit_flag(flag)
    if is_confirmed:
        ctx.deps.confirmed_flag = flag.strip()
    return display
