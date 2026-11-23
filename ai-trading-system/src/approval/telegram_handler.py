from __future__ import annotations

import asyncio
import logging
import json
import time
from typing import Optional, Callable, Awaitable

import aiohttp

logger = logging.getLogger(__name__)


class TelegramApprovalHandler:
    """
    Handles human approval via Telegram bot.
    
    When auto_execute is False, the orchestrator sends trade proposals
    to this handler, which posts them to Telegram. The human operator
    responds with /approve or /reject, and the handler returns the decision.
    """
    
    def __init__(self, bot_token: str, chat_id: str, timeout_seconds: int = 300) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout_seconds = timeout_seconds
        self._pending_decisions: dict[str, dict] = {}
        self._http_session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session
    
    async def send_approval(self, proposal: dict, symbol: str) -> Optional[str]:
        """
        Send a trade proposal to Telegram for approval.
        
        Args:
            proposal: TradeProposal dict
            symbol: Trading symbol
            
        Returns:
            "approve" or "reject" string if received within timeout, None otherwise
        """
        message_text = self._format_approval_message(proposal, symbol)
        message_id = await self._post_message(message_text)
        
        if message_id is None:
            return None
        
        self._pending_decisions[message_id] = {
            "symbol": symbol,
            "proposal": proposal,
            "timeout": time.time() + self.timeout_seconds,
        }
        
        # Wait for decision
        return await self._wait_for_decision(message_id)
    
    def _format_approval_message(self, proposal: dict, symbol: str) -> str:
        """Format the trade proposal for Telegram display."""
        side = proposal.get("action", "BUY")
        qty = proposal.get("quantity", 1)
        price = proposal.get("price", 0)
        symbol = proposal.get("symbol", symbol)
        order_value = proposal.get("order_value_inr", 0)
        
        emoji = "🟢" if side == "BUY" else "🔴"
        
        message = f"""{emoji} **Trade Approval**{emoji}
        
**Symbol:** {symbol}
**Action:** {side}
**Quantity:** {qty}
**Price:** {price:.2f}
**Order Value:** ₹{order_value:.2f}
**Time:** {time.strftime('%H:%M:%S', time.gmtime())}

**Reply with:**
• `/approve` - Execute trade
• `/reject` - Cancel trade"""
        
        return message
    
    async def _post_message(self, text: str) -> Optional[str]:
        """Post message to Telegram and return message_id."""
        try:
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if data.get("ok"):
                    return data["result"]["message_id"]
                else:
                    logger.error(f"Telegram error: {data}")
                    return None
        except Exception:
            logger.exception("Failed to post Telegram message")
            return None
    
    async def _wait_for_decision(self, message_id: str) -> Optional[str]:
        """Wait for approve/reject decision from human."""
        deadline = self._pending_decisions[message_id]["timeout"]
        
        while time.time() < deadline:
            # Check for new messages
            try:
                session = await self._get_session()
                url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
                payload = {
                    "offset": None,
                    "timeout": 10,
                }
                
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    
                    if data.get("ok"):
                        for update in data.get("result", []):
                            message_id_update = update["message"]["message_id"]
                            text = update["message"]["text"]
                            
                            if message_id_update == message_id:
                                if text == "/approve":
                                    return "approve"
                                elif text == "/reject":
                                    return "reject"
            except Exception:
                logger.exception("Error checking Telegram updates")
            
            await asyncio.sleep(2)
        
        # Timeout
        del self._pending_decisions[message_id]
        return None
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
    
    def is_pending(self, message_id: str) -> bool:
        """Check if a decision is still pending."""
        if message_id not in self._pending_decisions:
            return False
        return time.time() < self._pending_decisions[message_id]["timeout"]